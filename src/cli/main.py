"""
Rich terminal interface for the clinical intake agent.

Provides a polished CLI experience with color-coded conversation,
progress tracking, and formatted clinical brief output.
"""

import asyncio
import sys

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from src.agent.intake_agent import IntakeAgent
from src.models.clinical import ClinicalBrief
from src.models.conversation import (
    IntakeState,
    STATE_LABELS,
    STATE_ORDER,
)

# Console instance for all output.
console = Console()

# Color scheme.
AGENT_COLOR = "cyan"
PATIENT_COLOR = "green"
STATE_COLOR = "yellow"
HEADER_COLOR = "bold magenta"
ERROR_COLOR = "bold red"
MUTED_COLOR = "dim"
POSITIVE_COLOR = "red"
NEGATIVE_COLOR = "green"


def render_header() -> Panel:
    """Render the application header panel."""
    title = Text()
    title.append("  Clinical Intake Assistant  ", style="bold white on blue")
    title.append("  v0.1.0", style=MUTED_COLOR)
    return Panel(
        title,
        border_style="blue",
        padding=(0, 1),
    )


def render_progress(current_state: IntakeState) -> Panel:
    """
    Render the intake progress bar showing all phases.

    Args:
        current_state: The current IntakeState.

    Returns:
        A Rich Panel with the progress indicator.
    """
    current_index = STATE_ORDER.index(current_state)
    parts = Text()

    # Only show the main intake phases (skip COMPLETE).
    display_states = STATE_ORDER[:-1]

    for i, state in enumerate(display_states):
        label = STATE_LABELS[state]
        if i < current_index:
            # Completed phase.
            parts.append(f" [done] {label} ", style="bold green")
        elif i == current_index:
            # Current phase.
            parts.append(f" >> {label} << ", style="bold yellow")
        else:
            # Future phase.
            parts.append(f" [ ] {label} ", style=MUTED_COLOR)

        if i < len(display_states) - 1:
            parts.append(" -> ", style=MUTED_COLOR)

    return Panel(
        parts,
        title="[bold]Intake Progress[/bold]",
        border_style=STATE_COLOR,
        padding=(0, 1),
    )


def render_agent_message(content: str) -> None:
    """
    Display an agent message in the terminal.

    Args:
        content: The agent's message text.
    """
    console.print()
    console.print(
        Panel(
            Text(content),
            title="[bold cyan]Clinician[/bold cyan]",
            border_style=AGENT_COLOR,
            padding=(0, 2),
            width=min(console.width - 4, 80),
        )
    )


def render_patient_message(content: str) -> None:
    """
    Display a patient message in the terminal.

    Args:
        content: The patient's message text.
    """
    console.print(
        Panel(
            Text(content),
            title="[bold green]You (Patient)[/bold green]",
            border_style=PATIENT_COLOR,
            padding=(0, 2),
            width=min(console.width - 4, 80),
        )
    )


def render_state_change(new_state: IntakeState) -> None:
    """
    Display a state transition notification.

    Args:
        new_state: The new IntakeState.
    """
    label = STATE_LABELS.get(new_state, new_state.value)
    console.print()
    console.print(
        f"  --- Moving to: [bold yellow]{label}[/bold yellow] ---",
        justify="center",
    )
    console.print()


def render_clinical_brief(brief: ClinicalBrief) -> None:
    """
    Render the structured clinical brief as formatted Rich output.

    Displays the Chief Complaint, HPI (OLDCARTS table), ROS table,
    and additional notes in styled panels.

    Args:
        brief: The validated ClinicalBrief instance.
    """
    console.print()
    console.print()
    console.rule("[bold magenta] STRUCTURED CLINICAL BRIEF [/bold magenta]")
    console.print()

    # -- Chief Complaint --
    cc_text = Text()
    cc_text.append(brief.chief_complaint.statement)
    if brief.chief_complaint.duration:
        cc_text.append(f"\nDuration: {brief.chief_complaint.duration}", style=MUTED_COLOR)

    console.print(
        Panel(
            cc_text,
            title="[bold]Chief Complaint (CC)[/bold]",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # -- HPI (OLDCARTS) --
    hpi_table = Table(
        title="History of Present Illness (HPI) -- OLDCARTS",
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
        padding=(0, 2),
        expand=True,
    )
    hpi_table.add_column("Element", style="bold", width=22)
    hpi_table.add_column("Details", ratio=1)

    hpi = brief.hpi
    hpi_table.add_row("Onset", hpi.onset)
    hpi_table.add_row("Location", hpi.location)
    hpi_table.add_row("Duration", hpi.duration)
    hpi_table.add_row("Character", hpi.character)
    hpi_table.add_row("Aggravating Factors", hpi.aggravating_factors)
    hpi_table.add_row("Relieving Factors", hpi.relieving_factors)
    hpi_table.add_row("Timing", hpi.timing)
    hpi_table.add_row("Severity", hpi.severity)

    console.print()
    console.print(hpi_table)

    # -- Review of Systems --
    ros_table = Table(
        title="Review of Systems (ROS)",
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
        padding=(0, 2),
        expand=True,
    )
    ros_table.add_column("System", style="bold", width=22)
    ros_table.add_column("Finding", ratio=1)
    ros_table.add_column("+/-", width=6, justify="center")

    for finding in brief.ros:
        indicator_style = POSITIVE_COLOR if finding.is_positive else NEGATIVE_COLOR
        indicator = "+" if finding.is_positive else "-"
        ros_table.add_row(
            finding.system,
            finding.finding,
            Text(indicator, style=indicator_style),
        )

    console.print()
    console.print(ros_table)

    # -- Additional Notes --
    if brief.additional_notes and brief.additional_notes.strip():
        console.print()
        console.print(
            Panel(
                Text(brief.additional_notes),
                title="[bold]Additional Notes[/bold]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    # -- Timestamp --
    console.print()
    console.print(
        f"  Generated: {brief.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        style=MUTED_COLOR,
    )
    console.rule(style="magenta")
    console.print()


async def run_intake() -> None:
    """
    Main intake conversation loop.

    Initializes the agent, displays the greeting, and enters the
    conversation loop until the intake is complete or the user quits.
    """
    console.clear()
    console.print(render_header())
    console.print()

    # Initialize the agent.
    try:
        agent = IntakeAgent()
    except Exception as exc:
        console.print(
            f"[{ERROR_COLOR}]Failed to initialize agent: {exc}[/{ERROR_COLOR}]"
        )
        console.print(
            f"[{MUTED_COLOR}]Make sure OPENAI_API_KEY is set in your .env file.[/{MUTED_COLOR}]"
        )
        return

    console.print(render_progress(agent.get_current_state()))
    console.print()

    # Display the initial greeting.
    console.print(f"  [{MUTED_COLOR}]Connecting to AI...[/{MUTED_COLOR}]")
    try:
        greeting = await agent.get_greeting()
        render_agent_message(greeting)
    except Exception as exc:
        console.print(
            f"[{ERROR_COLOR}]Error getting greeting: {exc}[/{ERROR_COLOR}]"
        )
        return

    previous_state = agent.get_current_state()

    # Main conversation loop.
    while not agent.is_complete():
        console.print()

        # Get patient input.
        try:
            user_input = Prompt.ask(
                f"[bold {PATIENT_COLOR}]Patient[/bold {PATIENT_COLOR}]"
            )
        except (KeyboardInterrupt, EOFError):
            console.print(f"\n[{MUTED_COLOR}]Session ended by user.[/{MUTED_COLOR}]")
            return

        # Handle special commands.
        if user_input.strip().lower() == "/quit":
            console.print(f"[{MUTED_COLOR}]Session ended.[/{MUTED_COLOR}]")
            return

        if user_input.strip().lower() == "/state":
            console.print(render_progress(agent.get_current_state()))
            continue

        if user_input.strip().lower() == "/brief":
            console.print(
                f"  [{MUTED_COLOR}]Generating clinical brief...[/{MUTED_COLOR}]"
            )
            try:
                brief = await agent.generate_clinical_brief()
                render_clinical_brief(brief)
            except Exception as exc:
                console.print(
                    f"[{ERROR_COLOR}]Error generating brief: {exc}[/{ERROR_COLOR}]"
                )
            continue

        if not user_input.strip():
            continue

        # Display the patient's message.
        render_patient_message(user_input)

        # Process the message through the agent.
        console.print(f"  [{MUTED_COLOR}]...[/{MUTED_COLOR}]")
        try:
            response = await agent.process_message(user_input)
        except Exception as exc:
            console.print(
                f"[{ERROR_COLOR}]Error: {exc}[/{ERROR_COLOR}]"
            )
            continue

        # Check for state transition.
        current_state = agent.get_current_state()
        if current_state != previous_state:
            render_state_change(current_state)
            console.print(render_progress(current_state))
            previous_state = current_state

        # Display the agent's response.
        render_agent_message(response)

    # Intake is complete -- generate the clinical brief.
    console.print()
    console.print(
        f"  [{STATE_COLOR}]Intake complete. Generating clinical brief...[/{STATE_COLOR}]"
    )
    console.print()

    try:
        brief = await agent.generate_clinical_brief()
        render_clinical_brief(brief)
    except Exception as exc:
        console.print(
            f"[{ERROR_COLOR}]Error generating clinical brief: {exc}[/{ERROR_COLOR}]"
        )

    console.print(
        f"  [{MUTED_COLOR}]Thank you for using the Clinical Intake Assistant.[/{MUTED_COLOR}]"
    )


def run() -> None:
    """Entry point for the CLI application."""
    asyncio.run(run_intake())


if __name__ == "__main__":
    run()
