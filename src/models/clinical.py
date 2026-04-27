"""
Pydantic models for structured clinical output.

Defines the data structures for Chief Complaint, History of Present
Illness (using the OLDCARTS framework), Review of Systems findings,
and the final ClinicalBrief that encapsulates the full intake.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ChiefComplaint(BaseModel):
    """The primary reason for the patient's visit."""

    statement: str = Field(
        description="Concise statement of the chief complaint, e.g., "
        "'chest pain for 3 days'."
    )
    duration: Optional[str] = Field(
        default=None,
        description="How long the primary issue has been present, "
        "e.g., '3 days', '2 weeks'.",
    )


class HPIElement(BaseModel):
    """
    History of Present Illness structured using the OLDCARTS mnemonic.

    """

    onset: str = Field(
        description="When and how the symptom started. Include whether "
        "it was sudden or gradual and what the patient was doing."
    )
    location: str = Field(
        description="Where the symptom is located and whether it "
        "radiates to other areas."
    )
    duration: str = Field(
        description="How long each episode lasts, whether it is "
        "constant or intermittent."
    )
    character: str = Field(
        description="The nature/quality of the symptom, e.g., sharp, "
        "dull, aching, burning, throbbing, pressure-like."
    )
    aggravating_factors: str = Field(
        description="What makes the symptom worse, e.g., exertion, "
        "eating, certain positions."
    )
    relieving_factors: str = Field(
        description="What makes the symptom better, e.g., rest, "
        "medications, ice, heat."
    )
    timing: str = Field(
        description="Pattern of the symptom -- when it occurs, "
        "e.g., morning, after meals, constant, episodic."
    )
    severity: str = Field(
        description="Severity on a 1-10 scale, or qualitative "
        "description (mild, moderate, severe)."
    )


class ROSFinding(BaseModel):
    """A single Review of Systems finding."""

    system: str = Field(
        description="The organ system reviewed, e.g., 'Cardiovascular', "
        "'Respiratory', 'Gastrointestinal', 'Neurological'."
    )
    finding: str = Field(
        description="Description of the finding in that system."
    )
    is_positive: bool = Field(
        description="True if this is a pertinent positive (symptom "
        "present), False if pertinent negative (symptom denied)."
    )


class ClinicalBrief(BaseModel):
    """
    The complete structured clinical brief generated at the end of
    the intake conversation.

    Contains the chief complaint, HPI (OLDCARTS), review of systems
    findings, and any additional clinical notes.
    """

    chief_complaint: ChiefComplaint = Field(
        description="The patient's chief complaint."
    )
    hpi: HPIElement = Field(
        description="History of Present Illness structured using "
        "the OLDCARTS framework."
    )
    ros: list[ROSFinding] = Field(
        description="List of pertinent positive and negative Review "
        "of Systems findings across relevant organ systems."
    )
    additional_notes: str = Field(
        description="Any additional clinical observations, context, "
        "or notes not captured in the structured fields above."
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this clinical brief was generated.",
    )
