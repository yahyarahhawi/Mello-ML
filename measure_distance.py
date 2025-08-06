import json
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler

def load_book_embeddings_200():
    """Load the book embeddings from the 200-person dataset"""
    try:
        # Try the updated version first, then fall back to others
        filenames = [
            'book_embeddings_e5_200_randomized_updated.json',
            'book_embeddings_e5_200_randomized.json',
            'book_embeddings_e5_200.json'
        ]
        
        for filename in filenames:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                print(f"Loaded {len(profiles)} profiles from {filename}")
                return profiles
            except FileNotFoundError:
                continue
        
        raise FileNotFoundError("Could not find book embeddings file")
        
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

def find_user_by_name(profiles, name):
    """Find a user by name (case-insensitive)"""
    for profile in profiles:
        if profile.get('name', '').lower() == name.lower():
            return profile
    return None

def get_user_vector(profile):
    """Extract the books_vector from a user profile"""
    if 'books_vector' in profile:
        return np.array(profile['books_vector'])
    else:
        raise ValueError(f"No books_vector found for {profile.get('name', 'Unknown')}")

def calculate_distances(vector1, vector2, name1, name2):
    """Calculate various distance metrics between two vectors"""
    
    # Reshape vectors for calculations
    v1 = vector1.reshape(1, -1)
    v2 = vector2.reshape(1, -1)
    
    # 1. Cosine Similarity (higher = more similar)
    cosine_sim = cosine_similarity(v1, v2)[0][0]
    
    # 2. Euclidean Distance (lower = more similar)
    euclidean_dist = np.linalg.norm(vector1 - vector2)
    
    # 3. Manhattan Distance (L1 norm)
    manhattan_dist = np.sum(np.abs(vector1 - vector2))
    
    # 4. Cosine Distance (1 - cosine_similarity)
    cosine_dist = 1 - cosine_sim
    
    # 5. Normalized Euclidean Distance (using standardized vectors)
    scaler = StandardScaler()
    vectors_combined = np.vstack([vector1, vector2])
    vectors_scaled = scaler.fit_transform(vectors_combined)
    normalized_euclidean = np.linalg.norm(vectors_scaled[0] - vectors_scaled[1])
    
    return {
        'cosine_similarity': cosine_sim,
        'cosine_distance': cosine_dist,
        'euclidean_distance': euclidean_dist,
        'manhattan_distance': manhattan_dist,
        'normalized_euclidean': normalized_euclidean
    }

def format_distance_results(distances, name1, name2):
    """Format and display the distance results"""
    # Only output Euclidean distance
    euc_dist = distances['euclidean_distance']
    print(f"{euc_dist:.6f}")

def list_all_users(profiles):
    """List all available users"""
    print(f"\n{'='*80}")
    print("ALL AVAILABLE USERS")
    print(f"{'='*80}")
    
    # Sort users alphabetically
    names = [profile.get('name', 'Unknown') for profile in profiles]
    names.sort()
    
    for i, name in enumerate(names, 1):
        print(f"{i:3d}. {name}")
    
    print(f"\nTotal users: {len(names)}")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Measure distance between any two individuals by name',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python measure_distance.py "Yahya Rahhawi" "Michael Rodgers"
  python measure_distance.py "Yahya Rahhawi" "Veronica Armstrong"
  python measure_distance.py --list-users
        """
    )
    
    parser.add_argument('user1', nargs='?', help='Name of first user')
    parser.add_argument('user2', nargs='?', help='Name of second user')
    parser.add_argument('--list-users', action='store_true',
                       help='List all available users')
    
    args = parser.parse_args()
    
    print("Loading book embeddings from 200-person dataset...")
    profiles = load_book_embeddings_200()
    if not profiles:
        return
    
    if args.list_users:
        list_all_users(profiles)
        return
    
    if not args.user1 or not args.user2:
        print("Error: Please provide two user names")
        print("Example: python measure_distance.py 'Yahya Rahhawi' 'Michael Rodgers'")
        print("Use --list-users to see all available users")
        return
    
    # Find both users
    user1_profile = find_user_by_name(profiles, args.user1)
    user2_profile = find_user_by_name(profiles, args.user2)
    
    if user1_profile is None:
        print(f"Error: User '{args.user1}' not found")
        print("Use --list-users to see all available users")
        return
    
    if user2_profile is None:
        print(f"Error: User '{args.user2}' not found")
        print("Use --list-users to see all available users")
        return
    
    # Get their vectors
    try:
        vector1 = get_user_vector(user1_profile)
        vector2 = get_user_vector(user2_profile)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Calculate distances
    distances = calculate_distances(vector1, vector2, args.user1, args.user2)
    
    # Display results
    format_distance_results(distances, args.user1, args.user2)

if __name__ == "__main__":
    main() 