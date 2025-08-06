#!/usr/bin/env python3
"""
Add Lukka Wolff with custom profile and generate embedding.
"""

import json
import requests
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_gemini_embedding(text, api_key):
    """Generate embedding using Google Gemini API"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
    
    data = {
        "model": "models/text-embedding-004",
        "content": {
            "parts": [{"text": text}]
        },
        "taskType": "SEMANTIC_SIMILARITY"
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        embedding = result["embedding"]["values"]
        return embedding
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None

def main():
    print("👤 Adding Lukka Wolff with custom profile...")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in .env file")
        return
    
    # Load existing data with Yahya
    try:
        with open('book_gemini_embeddings_with_yahya.json', 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: book_gemini_embeddings_with_yahya.json not found.")
        return
    
    print(f"📊 Loaded {len(existing_data)} existing users")
    
    # Lukka's custom book taste profile
    lukka_profile = {
        "name": "Lukka Wolff",
        "book_taste": "They navigate the world with a mind that values both rigorous structure and fluid creativity. They likely find satisfaction in understanding how complex systems function, whether they are natural ecosystems or lines of code. This intellectual curiosity is balanced by a deep appreciation for beauty and expression, suggesting they see art and technology not as separate realms, but as interconnected tools for understanding and shaping reality. They appear to be a person who seeks to make a tangible, positive impact, driven by a blend of earnest purpose and innovative thinking. This desire to build, to create, and to contribute to something larger than themselves seems central to their inner compass.",
        "books_vector": None  # Will be filled by embedding generation
    }
    
    print(f"📝 Generating embedding for Lukka's book taste profile...")
    
    # Generate embedding for Lukka
    embedding = generate_gemini_embedding(lukka_profile["book_taste"], api_key)
    
    if embedding is None:
        print("❌ Failed to generate embedding for Lukka")
        return
    
    lukka_profile["books_vector"] = embedding
    print(f"✅ Generated {len(embedding)}-dimensional embedding for Lukka")
    
    # Combine all data
    final_data = existing_data + [lukka_profile]
    
    # Save combined data with Lukka
    with open('book_gemini_embeddings_with_yahya_and_lukka.json', 'w') as f:
        json.dump(final_data, f, indent=2)
    
    print(f"🎉 Successfully added Lukka to dataset with {len(final_data)} total users")
    print(f"📁 Saved to: book_gemini_embeddings_with_yahya_and_lukka.json")
    print(f"👤 Dataset now includes: Yahya Rahhawi + Lukka Wolff + 199 randomized users")

if __name__ == "__main__":
    main()