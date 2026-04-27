"""
Conversation state models for the clinical intake session.

Defines the intake state machine (IntakeState enum), individual
conversation messages, and the overall session container.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class IntakeState(str, Enum):
    """
    Represents the current phase of the clinical intake conversation.

    The agent progresses linearly through these states:
        GREETING -> CHIEF_COMPLAINT -> HPI_COLLECTION ->
        ROS_SCREENING -> SUMMARY -> COMPLETE
    """

    GREETING = "greeting"
    CHIEF_COMPLAINT = "chief_complaint"
    HPI_COLLECTION = "hpi_collection"
    ROS_SCREENING = "ros_screening"
    SUMMARY = "summary"
    COMPLETE = "complete"


# Ordered list of states for progress tracking.
STATE_ORDER: list[IntakeState] = [
    IntakeState.GREETING,
    IntakeState.CHIEF_COMPLAINT,
    IntakeState.HPI_COLLECTION,
    IntakeState.ROS_SCREENING,
    IntakeState.SUMMARY,
    IntakeState.COMPLETE,
]

# Human-readable labels for each state.
STATE_LABELS: dict[IntakeState, str] = {
    IntakeState.GREETING: "Greeting",
    IntakeState.CHIEF_COMPLAINT: "Chief Complaint",
    IntakeState.HPI_COLLECTION: "History of Present Illness",
    IntakeState.ROS_SCREENING: "Review of Systems",
    IntakeState.SUMMARY: "Summary & Confirmation",
    IntakeState.COMPLETE: "Complete",
}


class ConversationMessage(BaseModel):
    """A single message in the intake conversation."""

    role: str = Field(
        description="The sender role: 'assistant' (agent) or 'user' (patient)."
    )
    content: str = Field(
        description="The text content of the message."
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the message was sent.",
    )


class IntakeSession(BaseModel):
    """
    Container for a single clinical intake session.

    Tracks the conversation state, message history, and session
    metadata. All data is held in memory.
    """

    session_id: str = Field(
        default_factory=lambda: uuid4().hex[:12],
        description="Unique identifier for this intake session.",
    )
    state: IntakeState = Field(
        default=IntakeState.GREETING,
        description="Current phase of the intake conversation.",
    )
    messages: list[ConversationMessage] = Field(
        default_factory=list,
        description="Ordered list of all messages in the conversation.",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this session was created.",
    )

    def add_message(self, role: str, content: str) -> ConversationMessage:
        """
        Append a new message to the conversation history.

        Args:
            role: The sender role ('assistant' or 'user').
            content: The text content of the message.

        Returns:
            The newly created ConversationMessage.
        """
        message = ConversationMessage(role=role, content=content)
        self.messages.append(message)
        return message

    def advance_state(self) -> Optional[IntakeState]:
        """
        Move to the next state in the intake flow.

        Returns:
            The new IntakeState, or None if already at COMPLETE.
        """
        current_index = STATE_ORDER.index(self.state)
        if current_index < len(STATE_ORDER) - 1:
            self.state = STATE_ORDER[current_index + 1]
            return self.state
        return None

    def get_transcript(self) -> str:
        """
        Build a plain-text transcript of the full conversation.

        Returns:
            A formatted string with all messages labeled by role.
        """
        lines = []
        for msg in self.messages:
            role_label = "Clinician" if msg.role == "assistant" else "Patient"
            lines.append(f"{role_label}: {msg.content}")
        return "\n".join(lines)

    def get_state_index(self) -> int:
        """Return the zero-based index of the current state."""
        return STATE_ORDER.index(self.state)
