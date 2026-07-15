from typing import List, Literal
from pydantic import BaseModel, Field


class PatientProfile(BaseModel):
    age: int = Field(ge=18, le=100)
    sex: Literal["male", "female"]
    race: Literal["white", "black"] = "white"
    total_cholesterol: float = Field(ge=100, le=400)
    hdl_cholesterol: float = Field(ge=20, le=100)
    systolic_bp: float = Field(ge=90, le=250)
    smoker: bool = False
    diabetes: bool = False
    bp_meds: bool = False


class RiskAssessment(BaseModel):
    risk_percentage: float
    risk_tier: Literal["Low", "Borderline", "Intermediate", "High"]
    method: str
    out_of_validated_age_range: bool
    race_approximated_as_white: bool


class Citation(BaseModel):
    source: str
    page: int


class CarePlan(BaseModel):
    screening: List[str] = Field(description="Recommended screening tests and their cadence.")
    lifestyle: List[str] = Field(description="Lifestyle and behavioral interventions.")
    pharmacological: List[str] = Field(
        description="Medication considerations grounded in the guidelines; an empty list if none are indicated."
    )
    follow_up: List[str] = Field(description="Follow-up actions and their timing.")


class CritiqueResult(BaseModel):
    passes: bool = Field(
        description="True only if the plan has all four sections with substantive content, "
        "is grounded in the retrieved guideline excerpts, is tailored to the patient's risk "
        "tier, and contains no contraindicated suggestions."
    )
    feedback: str = Field(
        description="Specific, actionable feedback on what is missing or wrong. Empty string if passes is true."
    )
