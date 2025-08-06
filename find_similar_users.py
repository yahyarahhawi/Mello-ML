import json
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import sys

def load_embeddings(model_type):
    """Load embeddings based on model type"""
    if model_type.lower() == 'minilm':
        filename = 'semantic_profiles.json'
    elif model_type.lower() == 'e5':
        filename = 'semantic_profiles_e5.json'
    elif model_type.lower() == 'combined':
        filename = 'combined_profiles_e5.json'
    else:
        raise ValueError("Model must be 'minilm', 'e5', or 'combined'")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
        return profiles
    except FileNotFoundError:
        print(f"Error: {filename} not found. Run the appropriate embedding script first.")
        return None
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

def get_user_vector(profiles, name, category):
    """Get the vector for a specific user and category"""
    vector_key = f"{category}_vector"
    
    for profile in profiles:
        if profile['name'].lower() == name.lower():
            if vector_key in profile:
                return np.array(profile[vector_key]), profile['name']
            else:
                raise ValueError(f"Category '{category}' not found for user '{name}'")
    
    raise ValueError(f"User '{name}' not found")

def calculate_cosine_similarities(target_vector, profiles, category, use_mean_center=True):
    """Calculate cosine similarities between target user and all others"""
    vector_key = f"{category}_vector"
    
    # Extract all vectors
    all_vectors = []
    names = []
    
    for profile in profiles:
        if vector_key in profile:
            all_vectors.append(profile[vector_key])
            names.append(profile['name'])
    
    if not all_vectors:
        return []
    
    # Convert to numpy array
    embeddings = np.array(all_vectors)
    
    if use_mean_center:
        # Mean-center to highlight uniqueness
        mean_vector = np.mean(embeddings, axis=0)
        centered_embeddings = embeddings - mean_vector
        vectors_to_use = centered_embeddings
    else:
        # Use raw vectors
        vectors_to_use = embeddings
    
    # Find target vector index
    target_idx = None
    target_vector_flat = target_vector.flatten()
    for i, vector in enumerate(all_vectors):
        if np.allclose(vector, target_vector_flat):
            target_idx = i
            break
    
    if target_idx is None:
        raise ValueError("Target user not found in vectors")
    
    target_to_use = vectors_to_use[target_idx].reshape(1, -1)
    
    # Calculate similarities
    similarities = []
    for i, name in enumerate(names):
        vector = vectors_to_use[i].reshape(1, -1)
        similarity = cosine_similarity(target_to_use, vector)[0][0]
        similarities.append((name, similarity))
    
    return similarities

def calculate_nearest_neighbors(target_vector, profiles, category, use_mean_center=True, n_neighbors=None):
    """Calculate nearest neighbors using euclidean distance"""
    vector_key = f"{category}_vector"
    
    # Extract all vectors and names
    vectors = []
    names = []
    
    for profile in profiles:
        if vector_key in profile:
            vectors.append(profile[vector_key])
            names.append(profile['name'])
    
    if not vectors:
        return []
    
    # Convert to numpy array
    embeddings = np.array(vectors)
    
    if use_mean_center:
        # Mean-center to highlight uniqueness
        mean_vector = np.mean(embeddings, axis=0)
        vectors_to_use = embeddings - mean_vector
    else:
        # Use raw vectors
        vectors_to_use = embeddings
    
    # Find target user index
    target_vector_flat = target_vector.flatten()
    target_idx = None
    for i, vector in enumerate(vectors):
        if np.allclose(vector, target_vector_flat):
            target_idx = i
            break
    
    if target_idx is None:
        raise ValueError("Target user not found in vectors")
    
    # Set n_neighbors to all users if not specified
    if n_neighbors is None:
        n_neighbors = len(vectors_to_use)
    
    # Fit nearest neighbors
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(vectors_to_use)), metric='euclidean')
    nn.fit(vectors_to_use)
    
    # Find neighbors using target vector
    target_to_use = vectors_to_use[target_idx]
    distances, indices = nn.kneighbors([target_to_use])
    
    # Convert to similarities (smaller distance = higher similarity)
    # Using negative distance so we can sort in descending order
    similarities = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != target_idx:  # Exclude the target user themselves
            similarity = -dist  # Negative distance for sorting
            similarities.append((names[idx], similarity))
    
    return similarities

def format_output(similarities, target_name, category, method, model_type, use_mean_center=True):
    """Format and display the results"""
    print(f"\n{'='*60}")
    if use_mean_center:
        print(f"SIMILARITY SEARCH RESULTS (MEAN-CENTERED)")
    else:
        print(f"SIMILARITY SEARCH RESULTS (RAW VECTORS)")
    print(f"{'='*60}")
    print(f"Target User: {target_name}")
    print(f"Category: {category.title()}")
    print(f"Model: {model_type.upper()}")
    print(f"Method: {method.replace('_', ' ').title()}")
    if use_mean_center:
        print(f"Note: Vectors mean-centered to highlight uniqueness")
    else:
        print(f"Note: Using raw vectors without mean-centering")
    print(f"{'='*60}")
    
    if method == 'cosine_similarity':
        print(f"{'Rank':<4} {'User Name':<20} {'Cosine Similarity':<18}")
        print("-" * 44)
        for i, (name, score) in enumerate(similarities, 1):
            print(f"{i:<4} {name:<20} {score:.6f}")
    else:  # nearest_neighbor
        print(f"{'Rank':<4} {'User Name':<20} {'Distance (neg)':<15}")
        print("-" * 41)
        for i, (name, score) in enumerate(similarities, 1):
            print(f"{i:<4} {name:<20} {score:.6f}")
    
    print(f"\nTotal users ranked: {len(similarities)}")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Find users with similar taste based on semantic embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python find_similar_users.py "Elena Park" books minilm cosine_similarity
  python find_similar_users.py "Maya Ortiz" movies e5 nearest_neighbor
  python find_similar_users.py "Jordan Chen" music minilm cosine_similarity
        """
    )
    
    parser.add_argument('name', type=str, help='Name of the target user')
    parser.add_argument('category', choices=['books', 'movies', 'music'], 
                       help='Category to compare (books, movies, or music)')
    parser.add_argument('model', choices=['minilm', 'e5', 'combined'], 
                       help='Embedding model to use (minilm, e5, or combined)')
    parser.add_argument('method', choices=['cosine_similarity', 'nearest_neighbor'], 
                       help='Similarity calculation method')
    parser.add_argument('--no-mean-center', action='store_true',
                       help='Disable mean-centering (use raw vectors)')
    
    args = parser.parse_args()
    
    try:
        # Load embeddings
        print(f"Loading {args.model.upper()} embeddings...")
        profiles = load_embeddings(args.model)
        if not profiles:
            return
        
        # Get target user's vector
        print(f"Finding vector for user '{args.name}' in category '{args.category}'...")
        target_vector, actual_name = get_user_vector(profiles, args.name, args.category)
        
        # Calculate similarities
        use_mean_center = not args.no_mean_center
        print(f"Calculating similarities using {args.method.replace('_', ' ')}...")
        if args.method == 'cosine_similarity':
            similarities = calculate_cosine_similarities(target_vector, profiles, args.category, use_mean_center)
            # Remove target user and sort by similarity (descending)
            similarities = [(name, score) for name, score in similarities if name != actual_name]
            similarities.sort(key=lambda x: x[1], reverse=True)
        else:  # nearest_neighbor
            similarities = calculate_nearest_neighbors(target_vector, profiles, args.category, use_mean_center)
            # Sort by similarity (descending, which means smallest negative distance first)
            similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Display results
        format_output(similarities, actual_name, args.category, args.method, args.model, use_mean_center)
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()