from schemas.node_schema import AgentState
import logging
def guardrail_router(state: AgentState) -> str:
    """
    Router logic for determining the next node:

    Start Node:
        - "guardrail"

    Routing Rules:
        - If category == "rag":
              → route to "rag"
        - If category == "others":
              → ALWAYS route to "end" (intentional)
        - Otherwise:
              → If guardrail_pass is True  -> "valid"
              → If guardrail_pass is False -> "end"
    """
    logging.info("INSIDE GUARDRAIL ROUTER")
    messages = state.get("messages")
    if not messages:
        raise ValueError("No messages in state")

    last_msg = messages[-1]
    additional = getattr(last_msg, "additional_kwargs", {})
    guardrail_pass = additional.get("guardrail_pass", False)
    return "valid" if guardrail_pass else "end"

