#!/usr/bin/env python3
"""
Generate embeddings using Google Gemini embedding model from book_taste_profiles_200.json
"""

import os
import json
import requests
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

def generate_gemini_embedding(text, api_key):
    """Generate embedding using Google Gemini embedding model"""
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "models/text-embedding-004",
        "content": {
            "parts": [
                {
                    "text": text
                }
            ]
        },
        "taskType": "SEMANTIC_SIMILARITY"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        
        if 'embedding' in result and 'values' in result['embedding']:
            return result['embedding']['values']
        else:
            print(f"Unexpected response format: {result}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def main():
    print("🧠 Generating Gemini embeddings for book taste profiles...")
    
    # Load book taste profiles
    try:
        with open('book_taste_profiles_200.json', 'r') as f:
            profiles = json.load(f)
    except FileNotFoundError:
        print("❌ Error: book_taste_profiles_200.json not found.")
        return
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in environment variables.")
        return
    
    print(f"📊 Processing {len(profiles)} profiles...")
    
    embeddings_data = []
    
    for i, profile in enumerate(profiles):
        print(f"Processing {i+1}/{len(profiles)}: {profile['name']}")
        
        # Generate embedding for the book_taste text
        book_taste_text = profile['book_taste']
        
        # Add some rate limiting to avoid hitting API limits
        if i > 0:
            time.sleep(0.1)  # Small delay between requests
        
        embedding = generate_gemini_embedding(book_taste_text, api_key)
        
        if embedding:
            embedding_data = {
                "name": profile['name'],
                "book_taste": book_taste_text,
                "books_vector": embedding,  # Gemini embedding
                "books": profile.get('books', []),
                "personality_archetype": profile.get('personality_archetype', ''),
                "personality_description": profile.get('personality_description', '')
            }
            
            embeddings_data.append(embedding_data)
            print(f"  ✓ Generated {len(embedding)} dimensional embedding")
        else:
            print(f"  ✗ Failed to generate embedding for {profile['name']}")
    
    # Save embeddings
    output_file = "book_gemini_embeddings_200.json"
    with open(output_file, 'w') as f:
        json.dump(embeddings_data, f, indent=2)
    
    print(f"\n🎉 Generated embeddings for {len(embeddings_data)} users!")
    print(f"📁 Saved to: {output_file}")
    
    if len(embeddings_data) > 0:
        print(f"📊 Embedding dimensions: {len(embeddings_data[0]['books_vector'])}")
        
        # Check if Yahya is included
        yahya_found = any(user['name'] == 'Yahya Rahhawi' for user in embeddings_data)
        if yahya_found:
            print("✅ Yahya Rahhawi found in embeddings")
        else:
            print("⚠️ Yahya Rahhawi not found in embeddings")

if __name__ == "__main__":
    main()