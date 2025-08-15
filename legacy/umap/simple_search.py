#!/usr/bin/env python3
"""
Simple t-SNE search - just shows closest match
"""

import os
import json
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def generate_gemini_embedding(text, api_key):
    """Generate embedding using Google Gemini embedding model"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "models/text-embedding-004",
        "content": {"parts": [{"text": text}]},
        "taskType": "SEMANTIC_SIMILARITY"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        if 'embedding' in result and 'values' in result['embedding']:
            return result['embedding']['values']
        else:
            return None
            
    except Exception as e:
        return None

def find_closest_match(description):
    """Find the closest person to the given description"""
    # Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found")
        return
    
    # Load data
    try:
        with open('main.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ main.json not found")
        return
    
    # Get new embedding
    new_embedding = generate_gemini_embedding(description, api_key)
    if new_embedding is None:
        print("❌ Failed to generate embedding")
        return
    
    # Calculate similarities
    new_embedding = np.array(new_embedding)
    best_similarity = -1
    closest_person = None
    closest_description = None
    
    for entry in data:
        existing_embedding = np.array(entry['books_vector'])
        
        # Cosine similarity
        cosine_sim = np.dot(new_embedding, existing_embedding) / (
            np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding)
        )
        
        if cosine_sim > best_similarity:
            best_similarity = cosine_sim
            closest_person = entry.get('name', 'Unknown')
            closest_description = entry.get('book_taste', 'No description')
    
    print(f"🎯 Closest match: {closest_person}")
    print(f"📖 Their description: {closest_description}")
    print(f"📊 Similarity: {best_similarity:.4f}")

def main():
    # 🎯 EDIT THIS LINE:
    description = "someone who loves math, science, and the mysteries of the universe"
    
    find_closest_match(description)

if __name__ == "__main__":
    main()