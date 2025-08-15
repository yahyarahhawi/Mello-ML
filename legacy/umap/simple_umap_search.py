#!/usr/bin/env python3
"""
Simple UMAP search - clean output showing closest match
"""

import os
import json
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def generate_gemini_embedding(text, api_key):
    """Generate embedding using configured Gemini embedding model"""
    model = os.getenv('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-001')
    dimensions = int(os.getenv('GEMINI_EMBEDDING_DIMENSIONS', '768'))
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    data = {
        "model": f"models/{model}",
        "content": {"parts": [{"text": text}]},
        "taskType": "SEMANTIC_SIMILARITY",
        "outputDimensionality": dimensions
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
    """Find the closest person using UMAP embedding space"""
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
    
    # Calculate similarities (same as UMAP space uses cosine similarity)
    new_embedding = np.array(new_embedding)
    best_similarity = -1
    closest_person = None
    closest_description = None
    
    for entry in data:
        existing_embedding = np.array(entry['books_vector'])
        
        # Cosine similarity (UMAP uses cosine metric)
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