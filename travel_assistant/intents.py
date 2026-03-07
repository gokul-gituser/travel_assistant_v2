from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, model_validator





class Intent(str, Enum):
    INTENT_A_NEARBY_GENERIC = "INTENT_A_NEARBY_GENERIC"
    INTENT_B_NEARBY_BY_NEED = "INTENT_B_NEARBY_BY_NEED"
    INTENT_C_ITINERARY = "INTENT_C_ITINERARY"
    INTENT_D_FOOD_DIETARY = "INTENT_D_FOOD_DIETARY"
    INTENT_E_FRIENDS_BASED = "INTENT_E_FRIENDS_BASED"
    INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP = "INTENT_F_SAFETY_AND_PRACTICAL_TRAVEL_HELP"
    INTENT_G_URGENT_HEALTH = "INTENT_G_URGENT_HEALTH"
    INTENT_FALLBACK_GENERAL_TRAVEL = "INTENT_FALLBACK_GENERAL_TRAVEL"




class RoutingAction(str, Enum):
    ROUTE_INTENT = "ROUTE_INTENT"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    FALLBACK = "FALLBACK"
    URGENT_OVERRIDE = "URGENT_OVERRIDE"



# Intent Classification Result


class IntentClassificationResult(BaseModel):
    primary_intent: Intent = Field(description="The most likely intent")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0.0–1.0)")
    intent_scores: Dict[Intent, float] = Field(description="Scores for ALL intents")
    needs_clarification: bool = Field(default=False)
    clarification_reason: Optional[str] = Field(default=None)
    safety_override: bool = Field(default=False)

    @model_validator(mode="after")
    def validate_scores(self):
        # Primary intent must exist in intent_scores
        if self.primary_intent not in self.intent_scores:
            raise ValueError("primary_intent must exist in intent_scores")

        # Confidence must match primary intent score
        primary_score = self.intent_scores[self.primary_intent]
        if abs(primary_score - self.confidence) > 1e-6:
            raise ValueError(
                f"confidence ({self.confidence}) must equal "
                f"intent_scores[{self.primary_intent}] ({primary_score})"
            )

        # All intents must be present
        missing = set(Intent) - set(self.intent_scores.keys())
        if missing:
            raise ValueError(f"Missing intent scores for: {missing}")

        return self


class RouterDecision(BaseModel):
    """
    Output of the router.
    Determines the next node in the graph.
    """

    action: RoutingAction
    target_intent: Optional[Intent] = Field(
        default=None,
        description="Target intent when action == ROUTE_INTENT"
    )
