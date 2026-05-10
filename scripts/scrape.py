import json
import requests
import os

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
OUTPUT_PATH = "data/catalog.json"

def fetch_shl_catalog():
    print(f"Fetching actual JSON catalog from {CATALOG_URL}...")
    try:
        response = requests.get(CATALOG_URL, timeout=10)
        
        if response.status_code == 200:
            print("Success! Downloaded the live catalog. Parsing...")
            
            # THE FIX: strict=False allows unescaped control characters in the dirty SHL data
            raw_text = response.text
            try:
                catalog_data = json.loads(raw_text, strict=False)
                return catalog_data
            except json.JSONDecodeError as e:
                print(f"JSON Parse Error: {e}")
                return []
        else:
            print(f"Failed to fetch. Status code: {response.status_code}")
            return []

    except requests.exceptions.RequestException as e:
        print(f"Connection failed: {e}")
        return []

if __name__ == "__main__":
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    catalog_data = fetch_shl_catalog()
    
    # Save the exact JSON structure to our local data folder
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(catalog_data, f, indent=4)
        
    print(f"Successfully saved {len(catalog_data)} assessments to {OUTPUT_PATH}!")