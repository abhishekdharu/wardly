"""
Tests for the Pydantic clinical data models.

Validates serialization, deserialization, and field validation
for ChiefComplaint, HPIElement, ROSFinding, and ClinicalBrief.
"""

from datetime import datetime

import pytest

from src.models.clinical import (
    ChiefComplaint,
    ClinicalBrief,
    HPIElement,
    ROSFinding,
)


class TestChiefComplaint:
    """Tests for the ChiefComplaint model."""

    def test_valid_chief_complaint_with_duration(self):
        """A complete chief complaint with statement and duration."""
        cc = ChiefComplaint(
            statement="chest pain",
            duration="3 days",
        )
        assert cc.statement == "chest pain"
        assert cc.duration == "3 days"

    def test_valid_chief_complaint_without_duration(self):
        """Duration is optional and should default to None."""
        cc = ChiefComplaint(statement="headache")
        assert cc.statement == "headache"
        assert cc.duration is None

    def test_chief_complaint_requires_statement(self):
        """Statement is required -- omitting it should raise an error."""
        with pytest.raises(Exception):
            ChiefComplaint()


class TestHPIElement:
    """Tests for the HPIElement model (OLDCARTS)."""

    @pytest.fixture
    def valid_hpi_data(self) -> dict:
        """Fixture providing a valid set of HPI fields."""
        return {
            "onset": "Started suddenly 3 days ago while at rest",
            "location": "Central chest, radiating to left arm",
            "duration": "Each episode lasts 15-20 minutes",
            "character": "Pressure-like, squeezing",
            "aggravating_factors": "Exertion, stress",
            "relieving_factors": "Rest, sublingual nitroglycerin",
            "timing": "Intermittent, mostly in the morning",
            "severity": "7/10",
        }

    def test_valid_hpi(self, valid_hpi_data):
        """All OLDCARTS fields populated correctly."""
        hpi = HPIElement(**valid_hpi_data)
        assert hpi.onset == valid_hpi_data["onset"]
        assert hpi.severity == "7/10"

    def test_hpi_requires_all_fields(self):
        """All OLDCARTS fields are required."""
        with pytest.raises(Exception):
            HPIElement(onset="yesterday")

    def test_hpi_serialization_roundtrip(self, valid_hpi_data):
        """Model should survive JSON serialization and deserialization."""
        hpi = HPIElement(**valid_hpi_data)
        json_str = hpi.model_dump_json()
        restored = HPIElement.model_validate_json(json_str)
        assert restored == hpi


class TestROSFinding:
    """Tests for the ROSFinding model."""

    def test_positive_finding(self):
        """A pertinent positive ROS finding."""
        finding = ROSFinding(
            system="Cardiovascular",
            finding="Palpitations reported",
            is_positive=True,
        )
        assert finding.system == "Cardiovascular"
        assert finding.is_positive is True

    def test_negative_finding(self):
        """A pertinent negative ROS finding."""
        finding = ROSFinding(
            system="Respiratory",
            finding="Denies shortness of breath",
            is_positive=False,
        )
        assert finding.is_positive is False

    def test_finding_requires_all_fields(self):
        """All fields are required."""
        with pytest.raises(Exception):
            ROSFinding(system="Neurological")


class TestClinicalBrief:
    """Tests for the ClinicalBrief model."""

    @pytest.fixture
    def sample_brief_data(self) -> dict:
        """Fixture providing a complete clinical brief data set."""
        return {
            "chief_complaint": {
                "statement": "chest pain for 3 days",
                "duration": "3 days",
            },
            "hpi": {
                "onset": "Started 3 days ago at rest",
                "location": "Central chest",
                "duration": "15-20 minutes per episode",
                "character": "Pressure-like",
                "aggravating_factors": "Exertion",
                "relieving_factors": "Rest",
                "timing": "Intermittent, mornings",
                "severity": "7/10",
            },
            "ros": [
                {
                    "system": "Cardiovascular",
                    "finding": "Palpitations",
                    "is_positive": True,
                },
                {
                    "system": "Respiratory",
                    "finding": "No shortness of breath",
                    "is_positive": False,
                },
            ],
            "additional_notes": "Patient is anxious about symptoms.",
        }

    def test_valid_clinical_brief(self, sample_brief_data):
        """A complete clinical brief should validate successfully."""
        brief = ClinicalBrief(**sample_brief_data)
        assert brief.chief_complaint.statement == "chest pain for 3 days"
        assert len(brief.ros) == 2
        assert brief.timestamp is not None

    def test_brief_with_explicit_timestamp(self, sample_brief_data):
        """Providing an explicit timestamp should override the default."""
        ts = datetime(2026, 4, 27, 10, 0, 0)
        sample_brief_data["timestamp"] = ts
        brief = ClinicalBrief(**sample_brief_data)
        assert brief.timestamp == ts

    def test_brief_requires_all_sections(self):
        """Omitting required sections should raise an error."""
        with pytest.raises(Exception):
            ClinicalBrief(
                chief_complaint={"statement": "headache"},
            )

    def test_brief_empty_ros_list(self, sample_brief_data):
        """An empty ROS list should be valid (no findings)."""
        sample_brief_data["ros"] = []
        brief = ClinicalBrief(**sample_brief_data)
        assert brief.ros == []

    def test_brief_serialization_roundtrip(self, sample_brief_data):
        """Full brief should survive JSON roundtrip."""
        brief = ClinicalBrief(**sample_brief_data)
        json_str = brief.model_dump_json()
        restored = ClinicalBrief.model_validate_json(json_str)
        assert restored.chief_complaint == brief.chief_complaint
        assert restored.hpi == brief.hpi
        assert len(restored.ros) == len(brief.ros)
