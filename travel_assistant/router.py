from typing import Tuple

from config import AMBIGUITY_DELTA, HIGH_CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD
from intents import Intent, IntentClassificationResult, RouterDecision, RoutingAction

def route_intent(
    classification: IntentClassificationResult,
) -> RouterDecision:
    """
    Determine the next action based on intent classification result.
    """

    # 1. Safety override
    if classification.safety_override:
        return RouterDecision(
            action=RoutingAction.URGENT_OVERRIDE,
            target_intent=Intent.INTENT_G_URGENT_HEALTH,
        )

    # Sort intents by score (descending)
    sorted_scores: list[Tuple[Intent, float]] = sorted(
        classification.intent_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    

    top_intent, top_score = sorted_scores[0]
    second_intent, second_score = sorted_scores[1]
    gap = abs(top_score - second_score)

    # 2. Ambiguous intent → ask clarification
    if gap <= AMBIGUITY_DELTA:
        
        return RouterDecision(

            action=RoutingAction.ASK_CLARIFICATION,
            target_intent=None,
        )

    # 3. High confidence → route directly
    if top_score >= HIGH_CONFIDENCE_THRESHOLD:
        return RouterDecision(
            action=RoutingAction.ROUTE_INTENT,
            target_intent=top_intent,
        )

    

    # 4. Low confidence overall → fallback
    if top_score < LOW_CONFIDENCE_THRESHOLD:
        return RouterDecision(
            action=RoutingAction.FALLBACK,
            target_intent=Intent.INTENT_FALLBACK_GENERAL_TRAVEL,
        )

    # 5. Default: route to top intent
    return RouterDecision(
        action=RoutingAction.ROUTE_INTENT,
        target_intent=top_intent,
    )
