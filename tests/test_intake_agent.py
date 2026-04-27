"""
Tests for the IntakeAgent state machine and session management.

Uses mocked Azure OpenAI responses to test state transitions,
message handling, and conversation flow without making
actual API calls.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.intake_agent import IntakeAgent
from src.models.conversation import (
    IntakeSession,
    IntakeState,
    STATE_ORDER,
)


class TestIntakeSession:
    """Tests for the IntakeSession model and state machine."""

    def test_initial_state_is_greeting(self):
        """A new session should start in the GREETING state."""
        session = IntakeSession()
        assert session.state == IntakeState.GREETING

    def test_session_id_is_generated(self):
        """Session ID should be auto-generated and non-empty."""
        session = IntakeSession()
        assert len(session.session_id) == 12

    def test_add_message(self):
        """Adding a message should append to the messages list."""
        session = IntakeSession()
        msg = session.add_message(role="user", content="Hello")
        assert len(session.messages) == 1
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_advance_state_progresses_linearly(self):
        """State should advance through the defined order."""
        session = IntakeSession()
        expected_transitions = STATE_ORDER[1:]  # Skip GREETING (initial)

        for expected_state in expected_transitions:
            new_state = session.advance_state()
            assert new_state == expected_state

    def test_advance_state_returns_none_at_complete(self):
        """Advancing past COMPLETE should return None."""
        session = IntakeSession()
        # Advance to COMPLETE.
        for _ in range(len(STATE_ORDER) - 1):
            session.advance_state()

        assert session.state == IntakeState.COMPLETE
        result = session.advance_state()
        assert result is None

    def test_get_transcript(self):
        """Transcript should format messages with role labels."""
        session = IntakeSession()
        session.add_message("assistant", "Hello, how are you?")
        session.add_message("user", "I have a headache.")

        transcript = session.get_transcript()
        assert "Clinician: Hello, how are you?" in transcript
        assert "Patient: I have a headache." in transcript

    def test_get_state_index(self):
        """State index should correspond to position in STATE_ORDER."""
        session = IntakeSession()
        assert session.get_state_index() == 0

        session.advance_state()
        assert session.get_state_index() == 1


class TestIntakeAgentInit:
    """Tests for IntakeAgent initialization."""

    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_API_KEY": "test-key-123",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        },
    )
    def test_agent_creates_session(self):
        """Agent should create a new IntakeSession on init."""
        agent = IntakeAgent(model="gpt-4o-mini")
        assert agent.session is not None
        assert agent.session.state == IntakeState.GREETING

    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_API_KEY": "test-key-123",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        },
    )
    def test_agent_uses_provided_model(self):
        """Agent should use the model name passed to constructor."""
        agent = IntakeAgent(model="gpt-4o-mini")
        assert agent.model == "gpt-4o-mini"

    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_API_KEY": "test-key-123",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_MODEL": "gpt-4o",
        },
    )
    def test_agent_uses_env_model(self):
        """Agent should fall back to AZURE_OPENAI_MODEL env var."""
        agent = IntakeAgent()
        assert agent.model == "gpt-4o"

    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_API_KEY": "test-key-123",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        },
    )
    def test_is_complete_initially_false(self):
        """A freshly initialized agent should not be complete."""
        agent = IntakeAgent(model="gpt-4o-mini")
        assert agent.is_complete() is False

    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_API_KEY": "test-key-123",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        },
    )
    def test_get_current_state(self):
        """Current state should reflect the session state."""
        agent = IntakeAgent(model="gpt-4o-mini")
        assert agent.get_current_state() == IntakeState.GREETING


class TestIntakeAgentMessageProcessing:
    """Tests for message processing with mocked LLM calls."""

    @pytest.fixture
    def agent(self):
        """Create an agent with a mocked Azure OpenAI client."""
        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_API_KEY": "test-key-123",
                "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            },
        ):
            agent = IntakeAgent(model="gpt-4o-mini")
        return agent

    @pytest.mark.asyncio
    async def test_process_message_adds_user_message(self, agent):
        """Processing a message should record the user's input."""
        # Mock the LLM call to return a simple response.
        mock_message = MagicMock()
        mock_message.content = "I see, tell me more."
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        agent.client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        await agent.process_message("I have a headache")

        # Should have the user message and agent response.
        assert len(agent.session.messages) == 2
        assert agent.session.messages[0].role == "user"
        assert agent.session.messages[0].content == "I have a headache"
        assert agent.session.messages[1].role == "assistant"
        assert agent.session.messages[1].content == "I see, tell me more."

    @pytest.mark.asyncio
    async def test_process_message_returns_agent_response(self, agent):
        """Processing should return the agent's text response."""
        mock_message = MagicMock()
        mock_message.content = "What brings you in today?"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        agent.client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await agent.process_message("Hi")
        assert result == "What brings you in today?"

    @pytest.mark.asyncio
    async def test_get_greeting(self, agent):
        """Get greeting should return and record the agent's opening."""
        mock_message = MagicMock()
        mock_message.content = "Hello! I am your intake coordinator."
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        agent.client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        greeting = await agent.get_greeting()
        assert greeting == "Hello! I am your intake coordinator."
        assert len(agent.session.messages) == 1
        assert agent.session.messages[0].role == "assistant"
