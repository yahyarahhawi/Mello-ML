#!/usr/bin/env python3
"""
Simple script to find the closest person to a description
"""

import os
import json
import requests
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
    
    # Extract features and fit PCA
    features = []
    labels = []
    descriptions = []
    
    for entry in data:
        labels.append(entry.get('name', 'Unknown'))
        descriptions.append(entry.get('book_taste', 'No description'))
        features.append(entry['books_vector'])
    
    features = np.array(features)
    
    # Fit models
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=3)
    existing_coords = pca.fit_transform(features_scaled)
    
    # Transform new description
    embedding = generate_gemini_embedding(description, api_key)
    if embedding is None:
        print("❌ Failed to generate embedding")
        return
    
    new_features = np.array(embedding).reshape(1, -1)
    new_features_scaled = scaler.transform(new_features)
    new_coords = pca.transform(new_features_scaled)[0]
    
    # Find closest match
    distances = np.sqrt(np.sum((existing_coords - new_coords) ** 2, axis=1))
    closest_idx = np.argmin(distances)
    
    print(f"🎯 Closest match: {labels[closest_idx]}")
    print(f"📖 Their description: {descriptions[closest_idx]}")
    print(f"📏 Distance: {distances[closest_idx]:.4f}")

def main():
    # 🎯 EDIT THIS LINE:
    description = "someone who loves math, science, and the mysteries of the universe"
    
    find_closest_match(description)

if __name__ == "__main__":
    main()