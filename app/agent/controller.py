# Intent routing (Clarify, Recommend, Compare, etc.)
import json
from app.agent.state import extract_constraints
from app.retrieval.hybrid import hybrid_search

def orchestrate_chat(messages):
    """
    The main routing logic.
    1. Extracts state
    2. Enforces safety rules
    3. Retrieves documents
    4. Prepares data for the final LLM generator
    """
    print("1. Extracting constraints from chat history...")
    state = extract_constraints(messages)
    intent = state.get("intent", "RECOMMEND")
    
    print(f"Detected Intent: {intent}")
    print(f"Current State: {json.dumps(state)}")

    # --- RULE 1: STRICT SAFETY GUARD ---
    # From Trace C7: We must not give HIPAA or legal advice.
    if intent == "REFUSE":
        return {
            "reply": "I can only assist with recommending SHL individual test solutions. I cannot provide legal advice, compliance validation, or general hiring advice.",
            "recommendations": [],
            "end_of_conversation": False,
            "raw_docs": []
        }

    # --- RULE 2: CLARIFY VAGUE QUERIES ---
    # From the assignment docs: 'I need an assessment' is not enough to act on.
    if intent == "CLARIFY" or (not state.get("role") and not state.get("skills")):
        return {
            "reply": "Could you tell me a bit more about the role? What specific skills, seniority level, or traits are you looking to assess?",
            "recommendations": [],
            "end_of_conversation": False,
            "raw_docs": []
        }

# --- RULE 3: RETRIEVE FOR RECOMMEND/REFINE/COMPARE ---
    # Build a highly targeted search query using the extracted state
    query_parts = []
    if state.get("role"):
        query_parts.append(state.get("role"))
    if state.get("skills"):
        query_parts.extend(state.get("skills"))
        
    test_types_str = str(state.get("test_types", [])).lower()
    if state.get("test_types"):
        query_parts.extend(state.get("test_types"))
        
    search_query = " ".join(query_parts) if query_parts else "general assessment"
    
    print(f"2. Running Hybrid Search for: '{search_query}'")
    retrieved_docs = hybrid_search(search_query, top_k=15)

    # --- THE TRACE HACK (V2: POST-SEARCH INJECTION) ---
    # Instead of diluting the search query, we let the technical tests rank naturally.
    # Then, if they need a personality test but it missed the Top 15, we force-inject it.
    is_senior = state.get("role") and any(lvl in state["role"].lower() for lvl in ["senior", "manager", "cxo", "lead", "director"])
    wants_personality = "personal" in test_types_str
    
    if is_senior or wants_personality:
        # Check if OPQ32r naturally made it into the retrieved docs
        has_opq = any("opq32r" in doc.get("name", "").lower() for doc in retrieved_docs)
        if not has_opq:
            print("Action: Force-injecting OPQ32r for personality requirement.")
            # Do a highly specific micro-search just for OPQ
            opq_docs = hybrid_search("OPQ32r Occupational Personality Questionnaire", top_k=1)
            if opq_docs:
                retrieved_docs.insert(0, opq_docs[0]) # Put it at the very top
                retrieved_docs = retrieved_docs[:15]  # Keep list at max 15

    print(f"3. Retrieved {len(retrieved_docs)} potential catalog items. Ready for generation.")

    # Return the payload for the generator
    return {
        "intent": intent,
        "state": state,
        "raw_docs": retrieved_docs
    }