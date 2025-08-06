#!/usr/bin/env python3
"""
Generate E5 embeddings for the 200 book taste profiles.
Uses the multilingual E5-large model for semantic embeddings.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    print("🧠 Generating E5 embeddings for book taste profiles...")
    
    # Load taste profiles
    try:
        with open('book_taste_profiles_200.json', 'r') as f:
            profiles = json.load(f)
    except FileNotFoundError:
        print("❌ Error: book_taste_profiles_200.json not found. Run generate_book_taste_profiles.py first.")
        return
    
    # Load E5 model
    print("📥 Loading E5-large model...")
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    print("✅ Model loaded successfully")
    
    # Generate embeddings
    embeddings_data = []
    
    for i, profile in enumerate(profiles):
        print(f"Embedding {i+1}/200: {profile['name']}")
        
        # Check if this is Yahya - use existing embedding if available
        if profile['name'] == "Yahya Rahhawi":
            try:
                with open('combined_profiles_e5.json', 'r') as f:
                    existing_profiles = json.load(f)
                    for existing_profile in existing_profiles:
                        if existing_profile['name'] == "Yahya Rahhawi":
                            embedding_data = {
                                "name": profile['name'],
                                "book_taste": profile['book_taste'],
                                "books_vector": existing_profile['books_vector'],
                                "books": profile['books'],
                                "personality_archetype": profile.get('personality_archetype', ''),
                                "personality_description": profile.get('personality_description', '')
                            }
                            embeddings_data.append(embedding_data)
                            print(f"  ✓ Used existing embedding for Yahya ({len(existing_profile['books_vector'])} dimensions)")
                            break
                    else:
                        # Fallback: generate new embedding if not found
                        text_to_embed = f"passage: {profile['book_taste']}"
                        embedding = model.encode(text_to_embed, normalize_embeddings=True)
                        embedding_data = {
                            "name": profile['name'],
                            "book_taste": profile['book_taste'],
                            "books_vector": embedding.tolist(),
                            "books": profile['books'],
                            "personality_archetype": profile.get('personality_archetype', ''),
                            "personality_description": profile.get('personality_description', '')
                        }
                        embeddings_data.append(embedding_data)
                        print(f"  ✓ Generated new embedding for Yahya ({len(embedding)} dimensions)")
            except FileNotFoundError:
                # Generate new embedding if existing file not found
                text_to_embed = f"passage: {profile['book_taste']}"
                embedding = model.encode(text_to_embed, normalize_embeddings=True)
                embedding_data = {
                    "name": profile['name'],
                    "book_taste": profile['book_taste'],
                    "books_vector": embedding.tolist(),
                    "books": profile['books'],
                    "personality_archetype": profile.get('personality_archetype', ''),
                    "personality_description": profile.get('personality_description', '')
                }
                embeddings_data.append(embedding_data)
                print(f"  ✓ Generated new embedding for Yahya ({len(embedding)} dimensions)")
        else:
            # Generate new embedding for all other users
            text_to_embed = f"passage: {profile['book_taste']}"
            embedding = model.encode(text_to_embed, normalize_embeddings=True)
            
            embedding_data = {
                "name": profile['name'],
                "book_taste": profile['book_taste'],
                "books_vector": embedding.tolist(),  # Convert numpy array to list
                "books": profile['books'],
                "personality_archetype": profile.get('personality_archetype', ''),
                "personality_description": profile.get('personality_description', '')
            }
            
            embeddings_data.append(embedding_data)
            print(f"  ✓ Generated {len(embedding)} dimensional embedding")
    
    # Save embeddings
    output_file = "book_embeddings_e5_200.json"
    with open(output_file, 'w') as f:
        json.dump(embeddings_data, f, indent=2)
    
    print(f"\n🎉 Generated embeddings for {len(embeddings_data)} users!")
    print(f"📁 Saved to: {output_file}")
    print(f"📊 Embedding dimensions: {len(embeddings_data[0]['books_vector'])}")

if __name__ == "__main__":
    main()