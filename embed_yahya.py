#!/usr/bin/env python3
"""
Generate Gemini embedding for Yahya's profile and combine with randomized data.
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
    print("🧠 Generating Gemini embedding for Yahya...")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in .env file")
        return
    
    # Load Yahya's profile
    try:
        with open('yahya_profile.json', 'r') as f:
            yahya = json.load(f)
    except FileNotFoundError:
        print("❌ Error: yahya_profile.json not found.")
        return
    
    # Load randomized data
    try:
        with open('book_embeddings_randomized.json', 'r') as f:
            randomized_data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: book_embeddings_randomized.json not found.")
        return
    
    print(f"📝 Generating embedding for Yahya's book taste profile...")
    
    # Generate embedding for Yahya
    embedding = generate_gemini_embedding(yahya["book_taste"], api_key)
    
    if embedding is None:
        print("❌ Failed to generate embedding for Yahya")
        return
    
    yahya["books_vector"] = embedding
    print(f"✅ Generated {len(embedding)}-dimensional embedding for Yahya")
    
    # Combine all data
    final_data = randomized_data + [yahya]
    
    # Save combined data
    with open('book_gemini_embeddings_with_yahya.json', 'w') as f:
        json.dump(final_data, f, indent=2)
    
    print(f"🎉 Successfully created combined dataset with {len(final_data)} users")
    print(f"📁 Saved to: book_gemini_embeddings_with_yahya.json")
    print(f"👤 Yahya added as current user")

if __name__ == "__main__":
    main()