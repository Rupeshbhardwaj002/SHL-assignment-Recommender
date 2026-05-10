import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel('gemini-2.5-flash')

def generate_final_response(intent: str, state: dict, raw_docs: list, messages: list) -> dict:
    
    # 1. Format the conversation history
    history_text = ""
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else msg.role
        content = msg.get("content") if isinstance(msg, dict) else msg.content
        history_text += f"{role.upper()}: {content}\n"

    # 2. Format the retrieved documents (PATCHED WITH REAL SHL KEYS)
    catalog_context = "AVAILABLE CATALOG ITEMS:\n"
    if not raw_docs:
        catalog_context += "No items retrieved.\n"
    else:
        for i, doc in enumerate(raw_docs):
            # Map the real SHL data keys: 'link' and 'keys'
            item_url = doc.get('link', 'N/A')
            
            item_keys = doc.get('keys', [])
            item_type = ", ".join(item_keys) if isinstance(item_keys, list) else str(item_keys)
            
            catalog_context += f"[{i+1}] Name: {doc.get('name')} | URL: {item_url} | Type: {item_type} | Description: {doc.get('description', '')}\n"

    # 3. Build the ultra-strict System Prompt
    system_prompt = f"""
    You are an expert SHL Assessment Recommendation Agent.
    Your job is to recommend hiring assessments based on user needs.

    RULES FOR YOUR RESPONSE:
    1. NEVER invent or hallucinate assessment names or URLs.
    2. ONLY recommend assessments from the "AVAILABLE CATALOG ITEMS" provided below. Use the EXACT URL provided.
    3. If the intent is RECOMMEND or REFINE, provide 1 to 10 recommendations.
    4. If the intent is COMPARE, explain the difference between the requested tests using the catalog descriptions.
    5. Keep your 'reply' concise, professional, and directly address the user's constraints.
    6. 'end_of_conversation' should be true ONLY if the user explicitly confirms the final shortlist or says they are done. Otherwise, false.
    
    CURRENT INTENT: {intent}
    EXTRACTED USER CONSTRAINTS: {json.dumps(state)}
    
    {catalog_context}
    
    You MUST output valid JSON matching this exact schema:
    {{
        "reply": "Your conversational response explaining the recommendations or comparisons.",
        "recommendations": [
            {{"name": "Exact Name", "url": "Exact URL from catalog", "test_type": "Map to short code: K for Knowledge, P for Personality, A for Ability, C for Competencies, S for Simulations. Look at the Type field."}}
        ],
        "end_of_conversation": false
    }}
    Do not use markdown formatting like ```json in your response. Return pure JSON text.
    """

    prompt = system_prompt + "\n\nCONVERSATION HISTORY:\n" + history_text

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.0) # Zero temp for exact data matching
        )
        
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        final_json = json.loads(raw_text.strip())
        
        if "recommendations" in final_json and len(final_json["recommendations"]) > 10:
            final_json["recommendations"] = final_json["recommendations"][:10]
            
        return final_json
        
    except Exception as e:
        print(f"LLM Generation Error: {e}")
        return {
            "reply": "I encountered an error processing that request. Could you clarify your requirements?",
            "recommendations": [],
            "end_of_conversation": False
        }

# Quick local test
if __name__ == "__main__":
    test_intent = "RECOMMEND"
    test_state = {"role": "Java dev", "skills": ["Java"], "test_types": [], "intent": "RECOMMEND"}
    test_docs = [
        {"name": "Java 8 (New)", "url": "https://www.shl.com/java8", "test_type": "K", "description": "Tests Java 8."},
        {"name": "OPQ32r", "url": "https://www.shl.com/opq", "test_type": "P", "description": "Personality test."}
    ]
    test_messages = [{"role": "user", "content": "I need a Java test."}]
    
    print("Generating grounded response...")
    result = generate_final_response(test_intent, test_state, test_docs, test_messages)
    print(json.dumps(result, indent=4))