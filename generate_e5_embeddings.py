import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_taste_profiles():
    """Load taste profiles from JSON file"""
    try:
        with open('taste_profiles.json', 'r', encoding='utf-8') as f:
            profiles = json.load(f)
        return profiles
    except FileNotFoundError:
        print("Error: taste_profiles.json not found. Run generate_taste_profiles.py first.")
        return None
    except Exception as e:
        print(f"Error loading taste profiles: {e}")
        return None

def generate_embeddings(profiles):
    """Generate semantic embeddings using multilingual-e5-large model"""
    
    print("Loading multilingual-e5-large SentenceTransformer model...")
    print("Note: This is a large model (~2.5GB) and may take time to download on first use.")
    
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    
    semantic_profiles = []
    
    print(f"Generating E5 embeddings for {len(profiles)} users...")
    
    for i, profile in enumerate(profiles, 1):
        print(f"Processing user {i}/{len(profiles)}: {profile.get('name', 'Unknown')}")
        
        try:
            # Extract taste descriptions
            book_taste = profile.get('book_taste', '')
            movie_taste = profile.get('movie_taste', '')
            music_taste = profile.get('music_taste', '')
            
            # Add prefixes for E5 model (recommended for better performance)
            book_text = f"query: {book_taste}"
            movie_text = f"query: {movie_taste}"
            music_text = f"query: {music_taste}"
            
            # Generate embeddings
            book_embedding = model.encode(book_text, convert_to_tensor=True, normalize_embeddings=False)
            movie_embedding = model.encode(movie_text, convert_to_tensor=True, normalize_embeddings=False)
            music_embedding = model.encode(music_text, convert_to_tensor=True, normalize_embeddings=False)
            
            # Convert tensors to lists for JSON serialization
            semantic_profile = {
                "name": profile.get('name'),
                "books_vector": book_embedding.cpu().numpy().tolist(),
                "movies_vector": movie_embedding.cpu().numpy().tolist(),
                "music_vector": music_embedding.cpu().numpy().tolist()
            }
            
            semantic_profiles.append(semantic_profile)
            print(f"✓ Generated E5 embeddings for {profile.get('name')}")
            
        except Exception as e:
            print(f"✗ Failed to generate embeddings for {profile.get('name', 'Unknown')}: {e}")
    
    return semantic_profiles

def save_semantic_profiles(semantic_profiles):
    """Save semantic profiles to JSON file"""
    output_file = "semantic_profiles_e5.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(semantic_profiles, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Successfully saved {len(semantic_profiles)} E5 semantic profiles to {output_file}")
        return True
    except Exception as e:
        print(f"✗ Failed to save E5 semantic profiles: {e}")
        return False

def visualize_embeddings(semantic_profiles, category='books'):
    """Visualize E5 embeddings using PCA"""
    print(f"\nVisualizing E5 {category} embeddings with PCA...")
    
    # Extract embeddings for the specified category
    embeddings = []
    names = []
    
    vector_key = f"{category}_vector"
    for profile in semantic_profiles:
        if vector_key in profile:
            embeddings.append(profile[vector_key])
            names.append(profile['name'])
    
    if not embeddings:
        print(f"No embeddings found for category: {category}")
        return
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_array)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    # Add labels for each point
    for i, name in enumerate(names):
        plt.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    plt.title(f'{category.title()} Taste E5 Embeddings (PCA Visualization)')
    plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f'{category}_e5_embeddings_pca.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved E5 {category} embeddings visualization to {category}_e5_embeddings_pca.png")
    plt.close()

def compare_with_minilm():
    """Compare E5 and MiniLM embedding dimensions and properties"""
    try:
        # Load both embedding files
        with open('semantic_profiles.json', 'r') as f:
            minilm_profiles = json.load(f)
        with open('semantic_profiles_e5.json', 'r') as f:
            e5_profiles = json.load(f)
        
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        if minilm_profiles and e5_profiles:
            minilm_dims = len(minilm_profiles[0]['books_vector'])
            e5_dims = len(e5_profiles[0]['books_vector'])
            
            print(f"MiniLM-L6-v2 dimensions: {minilm_dims}")
            print(f"E5-Large dimensions: {e5_dims}")
            
            # Compare vector norms
            minilm_norm = np.linalg.norm(minilm_profiles[0]['books_vector'])
            e5_norm = np.linalg.norm(e5_profiles[0]['books_vector'])
            
            print(f"MiniLM sample vector norm: {minilm_norm:.4f}")
            print(f"E5 sample vector norm: {e5_norm:.4f}")
            
    except FileNotFoundError:
        print("Cannot compare - missing one of the embedding files")

def main():
    """Main function to generate E5 semantic embeddings"""
    
    # Load taste profiles
    profiles = load_taste_profiles()
    if not profiles:
        return
    
    # Generate E5 embeddings
    semantic_profiles = generate_embeddings(profiles)
    if not semantic_profiles:
        print("No E5 semantic profiles generated.")
        return
    
    # Save E5 semantic profiles
    if save_semantic_profiles(semantic_profiles):
        print(f"\n🎉 Successfully processed {len(semantic_profiles)} users with E5 embeddings!")
        
        # Generate visualizations
        print("\nGenerating E5 PCA visualizations...")
        visualize_embeddings(semantic_profiles, 'books')
        visualize_embeddings(semantic_profiles, 'movies')
        visualize_embeddings(semantic_profiles, 'music')
        
        # Print embedding dimensions
        if semantic_profiles:
            sample_embedding = semantic_profiles[0]['books_vector']
            print(f"\nE5 embedding dimensions: {len(sample_embedding)}")
            
        # Compare with MiniLM if available
        compare_with_minilm()
        
        print("\nReady for similarity matching with E5 embeddings!")

if __name__ == "__main__":
    main()