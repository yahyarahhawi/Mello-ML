import json
import numpy as np
import argparse
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

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

def get_archetype_groups(profiles):
    """Group users by their archetype"""
    archetype_groups = defaultdict(list)
    
    for profile in profiles:
        if 'personality_archetype' in profile and 'books_vector' in profile:
            archetype = profile['personality_archetype']
            archetype_groups[archetype].append(profile)
    
    return archetype_groups

def find_user_by_name(profiles, name):
    """Find a user by name (case-insensitive)"""
    for profile in profiles:
        if profile.get('name', '').lower() == name.lower():
            return profile
    return None

def calculate_manhattan_distance(vector1, vector2):
    """Calculate Manhattan distance between two vectors"""
    return np.sum(np.abs(vector1 - vector2))

def predict_archetype_for_user(target_vector, archetype_groups, method='nearest_neighbor'):
    """Predict archetype for a user based on Manhattan distance"""
    predictions = []
    
    for archetype, members in archetype_groups.items():
        if len(members) == 0:
            continue
            
        # Extract vectors for this archetype
        archetype_vectors = []
        for member in members:
            archetype_vectors.append(np.array(member['books_vector']))
        
        archetype_vectors = np.array(archetype_vectors)
        
        if method == 'nearest_neighbor':
            # Use nearest neighbor approach
            nn = NearestNeighbors(n_neighbors=min(3, len(archetype_vectors)), metric='manhattan')
            nn.fit(archetype_vectors)
            
            # Find nearest neighbors
            distances, indices = nn.kneighbors([target_vector])
            
            # Calculate average distance to nearest neighbors
            avg_distance = np.mean(distances[0])
            predictions.append((archetype, avg_distance, len(members)))
            
        elif method == 'average_distance':
            # Calculate average distance to all members of this archetype
            distances = []
            for vector in archetype_vectors:
                dist = calculate_manhattan_distance(target_vector, vector)
                distances.append(dist)
            
            avg_distance = np.mean(distances)
            predictions.append((archetype, avg_distance, len(members)))
            
        elif method == 'median_distance':
            # Calculate median distance to all members of this archetype
            distances = []
            for vector in archetype_vectors:
                dist = calculate_manhattan_distance(target_vector, vector)
                distances.append(dist)
            
            median_distance = np.median(distances)
            predictions.append((archetype, median_distance, len(members)))
    
    # Sort by distance (lower is better)
    predictions.sort(key=lambda x: x[1])
    return predictions

def format_prediction_results(predictions, target_name, method):
    """Format and display the prediction results"""
    print(f"\n{'='*80}")
    print(f"ARCHETYPE PREDICTION FOR: {target_name}")
    print(f"{'='*80}")
    print(f"Method: {method.replace('_', ' ').title()}")
    print(f"Distance metric: Manhattan distance")
    
    print(f"\nTop 10 Most Likely Archetypes:")
    print(f"{'Rank':<4} {'Archetype':<60} {'Distance':<12} {'Members':<8}")
    print("-" * 84)
    
    for i, (archetype, distance, member_count) in enumerate(predictions[:10], 1):
        archetype_short = archetype[:57] + "..." if len(archetype) > 60 else archetype
        print(f"{i:<4} {archetype_short:<60} {distance:.6f} {member_count:<8}")
    
    # Show confidence levels
    if len(predictions) > 0:
        best_archetype, best_distance, best_members = predictions[0]
        second_archetype, second_distance, second_members = predictions[1] if len(predictions) > 1 else (None, None, None)
        
        print(f"\nPrediction Summary:")
        print(f"🎯 Most likely archetype: {best_archetype}")
        print(f"   Distance: {best_distance:.6f}")
        print(f"   Members in this archetype: {best_members}")
        
        if second_archetype:
            distance_diff = second_distance - best_distance
            print(f"   Distance to 2nd choice: +{distance_diff:.6f}")
            
            # Calculate confidence based on distance difference
            if distance_diff > 1.0:
                confidence = "High"
            elif distance_diff > 0.5:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            print(f"   Confidence: {confidence}")
    
    return predictions

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
        description='Predict which archetype a person most likely belongs to based on Manhattan distance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_archetype.py "Yahya Rahhawi"
  python predict_archetype.py "Michael Rodgers" --method nearest_neighbor
  python predict_archetype.py "Veronica Armstrong" --method average_distance
  python predict_archetype.py --list-users
        """
    )
    
    parser.add_argument('user', nargs='?', help='Name of the user to predict archetype for')
    parser.add_argument('--method', choices=['nearest_neighbor', 'average_distance', 'median_distance'], 
                       default='nearest_neighbor',
                       help='Prediction method (default: nearest_neighbor)')
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
    
    if not args.user:
        print("Error: Please provide a user name")
        print("Example: python predict_archetype.py 'Yahya Rahhawi'")
        print("Use --list-users to see all available users")
        return
    
    # Find the target user
    target_profile = find_user_by_name(profiles, args.user)
    if target_profile is None:
        print(f"Error: User '{args.user}' not found")
        print("Use --list-users to see all available users")
        return
    
    # Get target user's vector
    target_vector = np.array(target_profile['books_vector'])
    target_name = target_profile['name']
    
    print(f"Found user: {target_name}")
    print(f"Vector dimensions: {len(target_vector)}")
    
    # Group users by archetype
    archetype_groups = get_archetype_groups(profiles)
    print(f"Found {len(archetype_groups)} unique archetypes")
    
    # Predict archetype
    predictions = predict_archetype_for_user(target_vector, archetype_groups, args.method)
    
    # Display results
    format_prediction_results(predictions, target_name, args.method)
    
    print(f"\n{'='*80}")
    print("PREDICTION COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 
