"""
Core intake agent orchestration.

The IntakeAgent manages the multi-turn clinical intake conversation,
using Azure OpenAI GPT-4o  with tool calling
for state transitions and structured output for clinical brief
generation.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from pydantic import ValidationError

from src.agent.prompts import (
    BASE_SYSTEM_PROMPT,
    BRIEF_GENERATION_PROMPT,
    STATE_INSTRUCTIONS,
    STATE_TRANSITION_TOOL,
)
from src.models.clinical import ClinicalBrief
from src.models.conversation import IntakeSession, IntakeState

# Load environment variables from .env file.
load_dotenv()

logger = logging.getLogger(__name__)


class IntakeAgent:
    """
    Orchestrates a clinical intake conversation with a patient.

    Uses a state machine to progress through intake phases and
    Azure OpenAI GPT-4o for natural language understanding and
    generation. State transitions are signaled by the LLM via
    tool calling.
    """

    def __init__(self, model: Optional[str] = None):
        """
        Initialize the intake agent.

        Args:
            model: Azure OpenAI deployment/model name. Defaults to
                   AZURE_OPENAI_MODEL env var or 'gpt-4o'.
        """
        self.client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_version="2024-12-01-preview",
        )
        self.model = model or os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
        self.session = IntakeSession()
        logger.info(
            "IntakeAgent initialized (session=%s, model=%s)",
            self.session.session_id,
            self.model,
        )

    async def process_message(self, user_input: str) -> str:
        """
        Process a patient message and return the agent's response.

        Handles the full cycle: appending the user message, calling
        the LLM, processing any tool calls (state transitions), and
        returning the assistant's text response.

        Args:
            user_input: The patient's message text.

        Returns:
            The agent's response text.
        """
        # Record the patient's message.
        self.session.add_message(role="user", content=user_input)

        # Build the message array and call the LLM.
        response_text = await self._call_llm()

        # Record the agent's response.
        self.session.add_message(role="assistant", content=response_text)

        return response_text

    async def get_greeting(self) -> str:
        """
        Generate the initial greeting message from the agent.

        Called once at the start of the session before any user input.

        Returns:
            The agent's opening greeting text.
        """
        response_text = await self._call_llm()
        self.session.add_message(role="assistant", content=response_text)
        return response_text

    async def generate_clinical_brief(self) -> ClinicalBrief:
        """
        Generate a structured clinical brief from the conversation.

        Uses OpenAI structured output (response_format) to extract
        the clinical information into a validated Pydantic model.

        Returns:
            A validated ClinicalBrief instance.

        Raises:
            ValueError: If brief generation fails after retries.
        """
        transcript = self.session.get_transcript()
        prompt = BRIEF_GENERATION_PROMPT.format(transcript=transcript)

        logger.info("Generating clinical brief from transcript")

        try:
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                ],
                response_format=ClinicalBrief,
            )

            brief = response.choices[0].message.parsed
            if brief is None:
                raise ValueError(
                    "Structured output parsing returned None. "
                    "The model may have refused the request."
                )

            # Override the LLM-generated timestamp with the actual
            # current time to prevent hallucinated dates.
            brief.timestamp = datetime.now()

            logger.info("Clinical brief generated successfully")
            return brief

        except ValidationError as exc:
            logger.error("Clinical brief validation failed: %s", exc)
            raise ValueError(
                f"Failed to generate a valid clinical brief: {exc}"
            ) from exc

    def get_current_state(self) -> IntakeState:
        """Return the current intake state."""
        return self.session.state

    def is_complete(self) -> bool:
        """Check if the intake conversation is complete."""
        return self.session.state == IntakeState.COMPLETE

    # -------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------

    async def _call_llm(self) -> str:
        """
        Build the message array and call the OpenAI API.

        Handles tool calls (state transitions) by advancing the
        state and recursively calling the LLM to get the next
        conversational response.

        Returns:
            The assistant's text response.
        """
        messages = self._build_messages()

        # Only offer the state transition tool if not already complete.
        tools = (
            [STATE_TRANSITION_TOOL]
            if self.session.state != IntakeState.COMPLETE
            else None
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=0.7,
        )

        choice = response.choices[0]
        message = choice.message

        # Check if the model wants to call the state transition tool.
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "advance_intake_state":
                    args = json.loads(tool_call.function.arguments)
                    reason = args.get("reason", "No reason provided")
                    old_state = self.session.state
                    new_state = self.session.advance_state()

                    logger.info(
                        "State transition: %s -> %s (reason: %s)",
                        old_state.value,
                        new_state.value if new_state else "None",
                        reason,
                    )

            # After processing tool calls, call the LLM again to get
            # the actual conversational response for the new state.
            # We include the tool call and result in the message history.
            messages.append(message.model_dump())
            for tool_call in message.tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(
                            {
                                "status": "success",
                                "new_state": self.session.state.value,
                            }
                        ),
                    }
                )

            # Add the updated state instruction.
            state_instruction = STATE_INSTRUCTIONS.get(
                self.session.state, ""
            )
            if state_instruction:
                messages.append(
                    {
                        "role": "system",
                        "content": state_instruction,
                    }
                )

            # Recursive call to get the actual text response.
            follow_up = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=(
                    [STATE_TRANSITION_TOOL]
                    if self.session.state != IntakeState.COMPLETE
                    else None
                ),
                temperature=0.7,
            )

            follow_up_message = follow_up.choices[0].message

            # Handle edge case: model tries to transition again
            # immediately (e.g., greeting -> CC in one shot).
            if follow_up_message.tool_calls:
                for tool_call in follow_up_message.tool_calls:
                    if tool_call.function.name == "advance_intake_state":
                        args = json.loads(tool_call.function.arguments)
                        reason = args.get("reason", "No reason provided")
                        old_state = self.session.state
                        new_state = self.session.advance_state()
                        logger.info(
                            "Double state transition: %s -> %s (reason: %s)",
                            old_state.value,
                            new_state.value if new_state else "None",
                            reason,
                        )

                # One more call to get actual text.
                messages.append(follow_up_message.model_dump())
                for tool_call in follow_up_message.tool_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(
                                {
                                    "status": "success",
                                    "new_state": self.session.state.value,
                                }
                            ),
                        }
                    )

                state_instruction = STATE_INSTRUCTIONS.get(
                    self.session.state, ""
                )
                if state_instruction:
                    messages.append(
                        {
                            "role": "system",
                            "content": state_instruction,
                        }
                    )

                final = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=(
                        [STATE_TRANSITION_TOOL]
                        if self.session.state != IntakeState.COMPLETE
                        else None
                    ),
                    temperature=0.7,
                )
                return final.choices[0].message.content or ""

            return follow_up_message.content or ""

        # No tool call -- return the text response directly.
        return message.content or ""

    def _build_messages(self) -> list[dict]:
        """
        Construct the message array for the OpenAI API call.

        Combines the base system prompt, current state instructions,
        and the full conversation history.

        Returns:
            A list of message dicts ready for the API.
        """
        messages = []

        # System prompt with current state context.
        state_instruction = STATE_INSTRUCTIONS.get(
            self.session.state, ""
        )
        system_content = f"{BASE_SYSTEM_PROMPT}\n\n{state_instruction}"
        messages.append({"role": "system", "content": system_content})

        # Conversation history.
        for msg in self.session.messages:
            messages.append(
                {"role": msg.role, "content": msg.content}
            )

        return messages
