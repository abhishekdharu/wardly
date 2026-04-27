# Wardly -- Pre-Visit Clinical Intake Agent

A CLI-based AI agent that conducts pre-visit clinical intake interviews with patients and generates structured clinical briefs. The agent systematically collects the **Chief Complaint (CC)**, **History of Present Illness (HPI)**  and **Review of Systems (ROS)**, then produces a formatted clinical summary ready for provider review.

## Features

- **Conversational intake** -- Natural, empathetic dialogue guided by a clinical state machine (not a rigid form)
- **HPI collection** -- Onset, Location, Duration, Character, Aggravating factors, Relieving factors, Timing, Severity
- **Targeted ROS screening** -- Screens 3-5 organ systems relevant to the chief complaint
- **Structured clinical brief** -- Auto-generated at the end using Pydantic structured output extraction
- **LLM-driven state transitions** -- The agent decides when enough information has been collected (no hardcoded rules)
- **Rich terminal UI** -- Color-coded conversation, progress bar, formatted tables for the clinical brief

## Architecture

```
+-----------------+       +-------------------+       +--------------------+
|                 |       |                   |       |                    |
|   Rich CLI      | ----> |   IntakeAgent     | ----> |  Azure OpenAI      |
|   (Terminal UI) |       |   (State Machine) |       |  GPT-4o            |
|                 | <---- |                   | <---- |                    |
+-----------------+       +---------+---------+       +--------------------+
                                    |
                          +---------+---------+
                          |  Clinical Brief   |
                          |  Generator        |
                          |  (Pydantic +      |
                          |   Structured      |
                          |   Output)         |
                          +-------------------+
```

### State Machine Flow

```
GREETING --> CHIEF_COMPLAINT --> HPI_COLLECTION --> ROS_SCREENING --> SUMMARY --> COMPLETE
```

The agent progresses linearly through each phase. State transitions are signaled by GPT-4o via **tool calling** -- the model calls `advance_intake_state` when it determines sufficient information has been collected for the current phase.

## Tech Stack

| Component          | Technology                                          |
|--------------------|-----------------------------------------------------|
| Language           | Python 3.11+                                        |
| Package Manager    | uv                                                  |
| LLM                | Azure OpenAI GPT-4o (via Azure AI Foundry)          |
| Data Validation    | Pydantic v2                                         |
| CLI UI             | Rich                                                |
| Testing            | pytest, pytest-asyncio                              |

## Project Structure

```
wardly/
  pyproject.toml                 # Project config, dependencies
  README.md
  src/
    models/
      clinical.py                # Pydantic models: CC, HPI, ROS, ClinicalBrief
      conversation.py            # IntakeState enum, session state machine
    agent/
      prompts.py                 # System prompts, state-specific instructions, tool schemas
      intake_agent.py            # Core agent orchestration (state machine + Azure OpenAI)
    cli/
      main.py                    # Rich terminal interface
  tests/
    test_models.py               # Pydantic model validation tests (14 tests)
    test_intake_agent.py         # Agent state transition and behavior tests (15 tests)
```

## Setup

### 1. Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- Azure OpenAI API access (via Azure AI Foundry)

### 2. Clone and install

```bash
cd wardly
uv sync
```

### 3. Configure environment

Edit `.env` with your Azure OpenAI credentials:

```env
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_MODEL=gpt-4o
```

### 4. Run the agent

```bash
uv run python -m src.cli.main
```

## Usage

Once running, the agent will greet you as **Sarah Mitchell, the intake coordinator**. Respond as the patient would. The conversation flows through:

1. **Greeting** -- Establishes rapport
2. **Chief Complaint** -- "What brings you in today?"
3. **HPI Collection** -- Naturally probes OLDCARTS elements
4. **ROS Screening** -- Screens relevant organ systems
5. **Summary** -- Reviews findings with the patient for confirmation

At the end, a **structured clinical brief** is generated and displayed with:
- Chief Complaint panel
- HPI table 
- ROS table 
- Additional notes

### CLI Commands

| Command   | Action                              |
|-----------|-------------------------------------|
| `/state`  | Show current intake progress        |
| `/brief`  | Force early brief generation        |
| `/quit`   | Exit the session                    |

## Running Tests

```bash
uv run pytest tests/ -v
```

All 29 tests run with mocked API calls (no Azure OpenAI key required for testing).

## Clinical Output Example

The generated clinical brief includes:

- **CC**: Concise chief complaint statement with duration
- **HPI (OLDCARTS)**: Structured table covering Onset, Location, Duration, Character, Aggravating Factors, Relieving Factors, Timing, and Severity
- **ROS**: Table of screened organ systems with pertinent positive (+) and negative (-) findings
- **Additional Notes**: Context not captured in structured fields
- **Timestamp**: Actual generation time
