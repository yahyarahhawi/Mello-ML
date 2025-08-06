#!/usr/bin/env python3
"""
Add personality embedding to existing embeddings file and generate PCA visualization.
Usage: python embed_personality.py "description" "Name" [input_json_file]
"""

import json
import requests
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import argparse
import shutil
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

def generate_pca_visualization(data):
    """Generate 3D PCA visualization from embeddings"""
    print("📊 Generating 3D PCA visualization...")
    
    # Extract embeddings and names - use books_vector for existing users, personality_vector for new user
    embeddings = []
    names = []
    
    for user in data:
        # Use personality_vector if available, otherwise books_vector
        if 'personality_vector' in user and user['personality_vector']:
            embeddings.append(user['personality_vector'])
            names.append(user['name'])
        elif 'books_vector' in user and user['books_vector']:
            embeddings.append(user['books_vector'])
            names.append(user['name'])
    
    if len(embeddings) < 2:
        print(f"⚠️ Only {len(embeddings)} embeddings found, need at least 2 for PCA")
        return None, None
    
    embeddings = np.array(embeddings)
    print(f"📊 Embeddings shape: {embeddings.shape}")
    
    # Perform PCA to 3D
    pca = PCA(n_components=min(3, len(embeddings)))
    embeddings_3d = pca.fit_transform(embeddings)
    
    print(f"📈 PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"📈 Total variance captured: {np.sum(pca.explained_variance_ratio_):.1%}")
    
    # Find the new user (has personality_vector)
    new_user_name = None
    for user in data:
        if 'personality_vector' in user and user['personality_vector']:
            new_user_name = user['name']
            break
    
    # Create visualization data
    users_3d = []
    for i, name in enumerate(names):
        user_data = {
            "name": name,
            "x": float(embeddings_3d[i, 0]),
            "y": float(embeddings_3d[i, 1]) if embeddings_3d.shape[1] > 1 else 0,
            "z": float(embeddings_3d[i, 2]) if embeddings_3d.shape[1] > 2 else 0,
            "isCurrentUser": name == new_user_name
        }
        users_3d.append(user_data)
    
    # Create PCA data structure
    pca_data = {
        "users": users_3d,
        "explained_variance": {
            "pc1": float(pca.explained_variance_ratio_[0]),
            "pc2": float(pca.explained_variance_ratio_[1]) if len(pca.explained_variance_ratio_) > 1 else 0,
            "pc3": float(pca.explained_variance_ratio_[2]) if len(pca.explained_variance_ratio_) > 2 else 0,
            "total": float(np.sum(pca.explained_variance_ratio_))
        },
        "statistics": {
            "total_users": len(users_3d),
            "dimensions_original": embeddings.shape[1],
            "dimensions_reduced": min(3, embeddings.shape[0]),
            "embedding_model": "Google Gemini text-embedding-004"
        }
    }
    
    # Save PCA data
    with open('main_PCA.json', 'w') as f:
        json.dump(pca_data, f, indent=2)
    
    # Create visualization plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    for user in users_3d:
        color = 'red' if user['isCurrentUser'] else 'blue'
        size = 50 if user['isCurrentUser'] else 30
        ax.scatter(user['x'], user['y'], user['z'], c=color, s=size, alpha=0.7)
        
        # Label the current user
        if user['isCurrentUser']:
            ax.text(user['x'], user['y'], user['z'], f"  {user['name']}", fontsize=9)
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)" if len(pca.explained_variance_ratio_) > 1 else "PC2")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)" if len(pca.explained_variance_ratio_) > 2 else "PC3")
    ax.set_title(f"3D PCA: Mixed Embeddings (Books + Personality)\n({pca_data['explained_variance']['total']:.1%} total variance explained)")
    
    plt.tight_layout()
    plt.savefig('main_PCA.png', dpi=300, bbox_inches='tight')
    
    return pca_data, 'main_PCA.png'

def main():
    parser = argparse.ArgumentParser(description='Add personality embedding to existing embeddings file')
    parser.add_argument('description', help='Personality description to embed')
    parser.add_argument('name', help='Name of the person')
    parser.add_argument('input_file', nargs='?', default='book_gemini_embeddings_with_yahya_and_lukka.json', help='Input JSON file (default: book_gemini_embeddings_with_yahya_and_lukka.json)')
    
    args = parser.parse_args()
    
    print(f"👤 Adding {args.name} with personality embedding...")
    
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in .env file")
        return 1
    
    # Load existing data
    try:
        with open(args.input_file, 'r') as f:
            existing_data = json.load(f)
        print(f"📊 Loaded {len(existing_data)} existing users from {args.input_file}")
    except FileNotFoundError:
        print(f"❌ Error: {args.input_file} not found")
        return 1
    
    # Generate embedding for personality
    print(f"🧠 Generating personality embedding for {args.name}...")
    personality_embedding = generate_gemini_embedding(args.description, api_key)
    
    if personality_embedding is None:
        print(f"❌ Failed to generate embedding for {args.name}")
        return 1
    
    print(f"✅ Generated {len(personality_embedding)}-dimensional embedding")
    
    # Create new user profile
    new_user = {
        "name": args.name,
        "book_taste": "",  # Empty since this is personality-based
        "books_vector": [],  # Empty since this is personality-based
        "personality_description": args.description,
        "personality_vector": personality_embedding
    }
    
    # Check if user already exists
    found_existing = False
    for i, user in enumerate(existing_data):
        if user['name'] == args.name:
            print(f"⚠️ User '{args.name}' already exists. Updating...")
            existing_data[i] = new_user
            found_existing = True
            break
    
    if not found_existing:
        existing_data.append(new_user)
    
    # Save as main.json
    with open('main.json', 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    print(f"💾 Saved main.json with {len(existing_data)} users")
    
    # Generate PCA visualization
    pca_data, png_filename = generate_pca_visualization(existing_data)
    
    if pca_data:
        # Copy to both directories
        mello_ml_pca = "/Users/yahyarahhawi/Developer/Mello/Mello-ML/main_PCA.json"
        mello_prototype_pca = "/Users/yahyarahhawi/Developer/Mello/Mello-prototype/src/data/main_PCA.json"
        
        # Copy PCA data to prototype
        os.makedirs(os.path.dirname(mello_prototype_pca), exist_ok=True)
        shutil.copy2('main_PCA.json', mello_prototype_pca)
        
        # Copy PNG files
        mello_ml_png = "/Users/yahyarahhawi/Developer/Mello/Mello-ML/main_PCA.png"
        mello_prototype_png = "/Users/yahyarahhawi/Developer/Mello/Mello-prototype/public/main_PCA.png"
        
        if os.path.exists(png_filename):
            os.makedirs(os.path.dirname(mello_prototype_png), exist_ok=True)
            shutil.copy2(png_filename, mello_prototype_png)
        
        print(f"🎉 Successfully processed {args.name}!")
        print(f"📁 Main data: main.json")
        print(f"📁 PCA data: {mello_ml_pca}")
        print(f"📁 PCA data (prototype): {mello_prototype_pca}")
        print(f"📁 Visualization: {mello_ml_png}")
        print(f"📁 Visualization (prototype): {mello_prototype_png}")
        print(f"📊 Total users: {len(existing_data)}")
        print(f"📊 PCA variance explained: {pca_data['explained_variance']['total']:.1%}")
    else:
        print(f"⚠️ PCA visualization skipped (insufficient data)")
        print(f"🎉 Successfully added {args.name} to main.json")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())