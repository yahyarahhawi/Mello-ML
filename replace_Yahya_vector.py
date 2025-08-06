import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

def load_e5_model():
    """Load the E5 embedding model"""
    print("Loading E5 embedding model...")
    try:
        # Load the E5 model - using the same model as in other scripts
        model = SentenceTransformer('intfloat/e5-large-v2')
        print("E5 model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading E5 model: {e}")
        return None

def embed_text_with_e5(text, model):
    """Embed text using E5 model"""
    print(f"Embedding text: {text[:100]}...")
    
    try:
        # E5 expects specific prefixes for different tasks
        # For book-related embeddings, we'll use the "query: " prefix
        formatted_text = f"query: {text}"
        
        # Generate embedding
        embedding = model.encode(formatted_text, convert_to_tensor=True)
        
        # Convert to numpy array and ensure it's the right shape
        embedding_np = embedding.cpu().numpy()
        
        print(f"Generated embedding with shape: {embedding_np.shape}")
        print(f"Embedding stats - Min: {embedding_np.min():.6f}, Max: {embedding_np.max():.6f}, Mean: {embedding_np.mean():.6f}")
        
        return embedding_np
        
    except Exception as e:
        print(f"Error embedding text: {e}")
        return None

def load_book_embeddings():
    """Load the book embeddings file"""
    try:
        filename = 'book_embeddings_e5_200_randomized.json'
        with open(filename, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
        print(f"Loaded {len(profiles)} profiles from {filename}")
        return profiles
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return None
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

def find_yahya_profile(profiles):
    """Find Yahya Rahhawi's profile in the data"""
    for i, profile in enumerate(profiles):
        if profile.get('name', '').lower() == 'yahya rahhawi':
            print(f"Found Yahya Rahhawi at index {i}")
            return i, profile
    
    print("Yahya Rahhawi not found in profiles")
    return None, None

def update_yahya_vector(profiles, new_vector):
    """Update Yahya's books_vector with the new embedding"""
    yahya_index, yahya_profile = find_yahya_profile(profiles)
    
    if yahya_index is None:
        print("Cannot update: Yahya Rahhawi not found")
        return False
    
    # Update the books_vector
    profiles[yahya_index]['books_vector'] = new_vector.tolist()
    
    print(f"Updated Yahya's books_vector with new embedding")
    print(f"New vector shape: {new_vector.shape}")
    print(f"New vector stats - Min: {new_vector.min():.6f}, Max: {new_vector.max():.6f}, Mean: {new_vector.mean():.6f}")
    
    return True

def save_updated_embeddings(profiles, output_filename=None):
    """Save the updated embeddings back to file"""
    if output_filename is None:
        output_filename = 'book_embeddings_e5_200_randomized_updated.json'
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
        print(f"Updated embeddings saved to: {output_filename}")
        return True
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        return False

def main():
    """Main function to replace Yahya's vector"""
    
    # The text to embed
    yahya_description = """This individual appears to be a deeply reflective and morally conscious person. He seeks out narratives that challenge conventional heroism and delve into the complexities of human motivation and societal influence. He values authenticity, a clear conscience, and a practical approach to doing good over grand, performative gestures. His reading suggests a keen interest in philosophical and existential questions, particularly regarding life's purpose and the nature of human suffering. He seems to appreciate profound emotional depth and a nuanced understanding of social dynamics, often drawn to stories that expose hypocrisy or explore the corrosive effects of certain environments. He is likely someone who grapples with serious ideas and strives for personal and intellectual growth."""
    
    print("="*80)
    print("REPLACING YAHYA'S BOOK VECTOR WITH NEW E5 EMBEDDING")
    print("="*80)
    
    # Step 1: Load E5 model
    model = load_e5_model()
    if model is None:
        return
    
    # Step 2: Embed the text
    new_vector = embed_text_with_e5(yahya_description, model)
    if new_vector is None:
        return
    
    # Step 3: Load current embeddings
    profiles = load_book_embeddings()
    if profiles is None:
        return
    
    # Step 4: Find and update Yahya's profile
    success = update_yahya_vector(profiles, new_vector)
    if not success:
        return
    
    # Step 5: Save updated embeddings
    save_updated_embeddings(profiles)
    
    print("\n" + "="*80)
    print("PROCESS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("Yahya's books_vector has been updated with the new E5 embedding.")
    print("The updated file contains the new vector that reflects his described personality and reading preferences.")

if __name__ == "__main__":
    main() 