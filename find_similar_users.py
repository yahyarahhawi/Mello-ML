import json
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def load_book_embeddings_200():
    """Load the book embeddings from the 200-person dataset with Gemini vectors"""
    try:
        # Load the new Gemini embeddings with Yahya
        filename = 'main.json'
        with open(filename, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
        print(f"Loaded {len(profiles)} profiles from {filename}")
        return profiles
        
    except FileNotFoundError:
        print(f"Error: {filename} not found. Make sure to generate it first.")
        return None
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

def get_user_vector(profiles, target_name):
    """Get the specified user's book vector"""
    for profile in profiles:
        if profile['name'].lower() == target_name.lower():
            if 'books_vector' in profile:
                return np.array(profile['books_vector']), profile['name']
            else:
                raise ValueError(f"books_vector not found for {target_name}")
    
    raise ValueError(f"{target_name} not found in profiles")

def calculate_manhattan_similarities(target_vector, profiles, target_name, use_mean_center=True):
    """Calculate Manhattan distance similarities between target user and all others"""
    # Extract all book vectors
    all_vectors = []
    names = []
    
    for profile in profiles:
        if 'books_vector' in profile:
            all_vectors.append(profile['books_vector'])
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
    
    # Find target user's vector index
    target_idx = None
    target_vector_flat = target_vector.flatten()
    for i, vector in enumerate(all_vectors):
        if np.allclose(vector, target_vector_flat):
            target_idx = i
            break
    
    if target_idx is None:
        raise ValueError(f"{target_name} not found in vectors")
    
    target_to_use = vectors_to_use[target_idx]
    
    # Calculate Manhattan distances (lower = more similar)
    similarities = []
    for i, name in enumerate(names):
        if name.lower() != target_name.lower():  # Exclude target user themselves
            vector = vectors_to_use[i]
            # Manhattan distance is sum of absolute differences
            manhattan_dist = np.sum(np.abs(target_to_use - vector))
            # Convert to similarity (negative distance so lower distance = higher similarity)
            similarity = -manhattan_dist
            similarities.append((name, similarity))
    
    return similarities

def calculate_manhattan_neighbors(target_vector, profiles, target_name, use_mean_center=True, n_neighbors=10):
    """Calculate nearest neighbors using Manhattan distance"""
    # Extract all book vectors and names
    vectors = []
    names = []
    
    for profile in profiles:
        if 'books_vector' in profile:
            vectors.append(profile['books_vector'])
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
    
    # Find target user's index
    target_vector_flat = target_vector.flatten()
    target_idx = None
    for i, vector in enumerate(vectors):
        if np.allclose(vector, target_vector_flat):
            target_idx = i
            break
    
    if target_idx is None:
        raise ValueError(f"{target_name} not found in vectors")
    
    # Fit nearest neighbors with Manhattan distance
    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(vectors_to_use)), metric='manhattan')
    nn.fit(vectors_to_use)
    
    # Find neighbors using target user's vector
    target_to_use = vectors_to_use[target_idx]
    distances, indices = nn.kneighbors([target_to_use])
    
    # Convert to similarities (smaller distance = higher similarity)
    # Using negative distance so we can sort in descending order
    similarities = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != target_idx:  # Exclude target user themselves
            similarity = -dist  # Negative distance for sorting
            similarities.append((names[idx], similarity))
    
    return similarities

def format_output(similarities, target_name, method, use_mean_center=True, top_n=10, profiles=None):
    """Format and display the results"""
    print(f"\n{'='*70}")
    print(f"SIMILARITY SEARCH RESULTS (RAW VECTORS)")
    print(f"{'='*70}")
    print(f"Target User: {target_name}")
    print(f"Category: Books (768-dimensional Gemini embeddings)")
    print(f"Method: {method.replace('_', ' ').title()}")
    print(f"Note: Using raw vectors without mean-centering")
    print(f"{'='*70}")
    
    # Show top N results
    top_similarities = similarities[:top_n]
    
    if method == 'manhattan_similarity':
        print(f"{'Rank':<4} {'User Name':<25} {'Manhattan Similarity':<18} {'Archetype':<40}")
        print("-" * 89)
        for i, (name, score) in enumerate(top_similarities, 1):
            # Find archetype for this user
            archetype = "Unknown"
            if profiles:
                for profile in profiles:
                    if profile.get('name', '').lower() == name.lower():
                        archetype = profile.get('personality_archetype', 'Unknown')
                        break
            
            # Truncate archetype if too long
            archetype_short = archetype[:37] + "..." if len(archetype) > 40 else archetype
            print(f"{i:<4} {name:<25} {score:.6f} {archetype_short:<40}")
    else:  # manhattan_neighbor
        print(f"{'Rank':<4} {'User Name':<25} {'Manhattan Distance (neg)':<20} {'Archetype':<40}")
        print("-" * 91)
        for i, (name, score) in enumerate(top_similarities, 1):
            # Find archetype for this user
            archetype = "Unknown"
            if profiles:
                for profile in profiles:
                    if profile.get('name', '').lower() == name.lower():
                        archetype = profile.get('personality_archetype', 'Unknown')
                        break
            
            # Truncate archetype if too long
            archetype_short = archetype[:37] + "..." if len(archetype) > 40 else archetype
            print(f"{i:<4} {name:<25} {score:.6f} {archetype_short:<40}")
    
    print(f"\nTop {len(top_similarities)} most similar users shown out of {len(similarities)} total")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Find users with similar book taste based on 768-dimensional Gemini embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python find_similar_users_200.py
  python find_similar_users_200.py "Michael Rodgers"
  python find_similar_users_200.py "Yahya Rahhawi"
        """
    )
    
    parser.add_argument('user', nargs='?', default='Yahya Rahhawi', 
                       help='Name of the target user (default: Yahya Rahhawi)')
    
    args = parser.parse_args()
    target_user = args.user
    
    print("Loading book embeddings from 200-person dataset...")
    profiles = load_book_embeddings_200()
    if not profiles:
        return
    
    try:
        # Get target user's vector
        print(f"Finding {target_user}'s book vector...")
        target_vector, actual_name = get_user_vector(profiles, target_user)
        print(f"Found vector for: {actual_name}")
        
        # Calculate similarities using both methods
        print("\n" + "="*70)
        print(f"ANALYSIS RESULTS FOR {actual_name.upper()}")
        print("="*70)
        
        # Manhattan Similarity (raw vectors only)
        print("\nMANHATTAN SIMILARITY (RAW VECTORS)")
        print("-" * 40)
        similarities_manhattan = calculate_manhattan_similarities(target_vector, profiles, target_user, use_mean_center=False)
        similarities_manhattan.sort(key=lambda x: x[1], reverse=True)
        format_output(similarities_manhattan, actual_name, 'manhattan_similarity', use_mean_center=False, top_n=10, profiles=profiles)
        
        print(f"\n{'='*70}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*70}")
        
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main() 
