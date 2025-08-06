import json
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import itertools

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

def get_archetype_groups(profiles):
    """Group users by their archetype"""
    archetype_groups = defaultdict(list)
    
    for profile in profiles:
        if 'personality_archetype' in profile and 'books_vector' in profile:
            archetype = profile['personality_archetype']
            archetype_groups[archetype].append(profile)
    
    return archetype_groups

def calculate_metrics_between_vectors(vector1, vector2):
    """Calculate various distance metrics between two vectors"""
    v1 = vector1.reshape(1, -1)
    v2 = vector2.reshape(1, -1)
    
    # Cosine similarity (higher = more similar)
    cosine_sim = cosine_similarity(v1, v2)[0][0]
    
    # Euclidean distance (lower = more similar)
    euclidean_dist = np.linalg.norm(vector1 - vector2)
    
    # Manhattan distance (lower = more similar)
    manhattan_dist = np.sum(np.abs(vector1 - vector2))
    
    # Cosine distance (lower = more similar)
    cosine_dist = 1 - cosine_sim
    
    # Normalized Euclidean distance
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

def analyze_archetype_metric_performance(archetype_groups):
    """Analyze which metrics best separate each archetype"""
    metrics = ['cosine_similarity', 'cosine_distance', 'euclidean_distance', 'manhattan_distance', 'normalized_euclidean']
    results = {}
    
    print("Analyzing metric performance for each archetype...")
    print("="*80)
    
    for archetype, users in archetype_groups.items():
        if len(users) < 2:  # Skip archetypes with only one user
            continue
            
        print(f"\nAnalyzing: {archetype}")
        print(f"Users in this archetype: {len(users)}")
        
        # Calculate within-archetype distances
        within_distances = defaultdict(list)
        within_pairs = list(itertools.combinations(users, 2))
        
        for user1, user2 in within_pairs:
            vector1 = np.array(user1['books_vector'])
            vector2 = np.array(user2['books_vector'])
            metrics_result = calculate_metrics_between_vectors(vector1, vector2)
            
            for metric in metrics:
                within_distances[metric].append(metrics_result[metric])
        
        # Calculate between-archetype distances
        between_distances = defaultdict(list)
        other_users = []
        for other_archetype, other_users_list in archetype_groups.items():
            if other_archetype != archetype:
                other_users.extend(other_users_list)
        
        # Sample some between-archetype pairs to avoid too many calculations
        within_users = users
        sample_size = min(50, len(within_users) * len(other_users) // 100)  # Sample 1% of possible pairs
        
        for _ in range(sample_size):
            user1 = np.random.choice(within_users)
            user2 = np.random.choice(other_users)
            vector1 = np.array(user1['books_vector'])
            vector2 = np.array(user2['books_vector'])
            metrics_result = calculate_metrics_between_vectors(vector1, vector2)
            
            for metric in metrics:
                between_distances[metric].append(metrics_result[metric])
        
        # Calculate separation scores for each metric
        separation_scores = {}
        for metric in metrics:
            if metric == 'cosine_similarity':
                # For cosine similarity, we want within > between
                within_mean = np.mean(within_distances[metric])
                between_mean = np.mean(between_distances[metric])
                separation_score = within_mean - between_mean
            else:
                # For distances, we want within < between
                within_mean = np.mean(within_distances[metric])
                between_mean = np.mean(between_distances[metric])
                separation_score = between_mean - within_mean
            
            separation_scores[metric] = separation_score
        
        # Find the best metric for this archetype
        best_metric = max(separation_scores.keys(), key=lambda x: separation_scores[x])
        
        results[archetype] = {
            'best_metric': best_metric,
            'separation_score': separation_scores[best_metric],
            'all_scores': separation_scores,
            'within_mean': {m: np.mean(within_distances[m]) for m in metrics},
            'between_mean': {m: np.mean(between_distances[m]) for m in metrics},
            'user_count': len(users)
        }
        
        print(f"  Best metric: {best_metric}")
        print(f"  Separation score: {separation_scores[best_metric]:.6f}")
        print(f"  Within-mean: {results[archetype]['within_mean'][best_metric]:.6f}")
        print(f"  Between-mean: {results[archetype]['between_mean'][best_metric]:.6f}")
    
    return results

def display_results(results):
    """Display the analysis results"""
    print("\n" + "="*80)
    print("ARCHETYPE METRIC ANALYSIS RESULTS")
    print("="*80)
    
    # Sort archetypes by separation score
    sorted_archetypes = sorted(results.items(), key=lambda x: x[1]['separation_score'], reverse=True)
    
    print(f"\n{'Rank':<4} {'Archetype':<50} {'Best Metric':<20} {'Score':<10}")
    print("-" * 84)
    
    for i, (archetype, data) in enumerate(sorted_archetypes, 1):
        archetype_short = archetype[:47] + "..." if len(archetype) > 50 else archetype
        print(f"{i:<4} {archetype_short:<50} {data['best_metric']:<20} {data['separation_score']:.6f}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    metric_counts = Counter([data['best_metric'] for data in results.values()])
    print(f"\nMost effective metrics across all archetypes:")
    for metric, count in metric_counts.most_common():
        print(f"  {metric}: {count} archetypes")
    
    # Show top 5 most separable archetypes
    print(f"\nTop 5 most separable archetypes:")
    for i, (archetype, data) in enumerate(sorted_archetypes[:5], 1):
        print(f"{i}. {archetype}")
        print(f"   Best metric: {data['best_metric']}")
        print(f"   Separation score: {data['separation_score']:.6f}")
        print(f"   Users: {data['user_count']}")
        print()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Analyze which distance metrics best separate archetypes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_archetype_metrics.py
        """
    )
    
    args = parser.parse_args()
    
    print("Loading book embeddings from 200-person dataset...")
    profiles = load_book_embeddings_200()
    if not profiles:
        return
    
    # Group users by archetype
    archetype_groups = get_archetype_groups(profiles)
    
    print(f"Found {len(archetype_groups)} unique archetypes")
    print(f"Total users with archetypes: {sum(len(users) for users in archetype_groups.values())}")
    
    # Analyze metric performance
    results = analyze_archetype_metric_performance(archetype_groups)
    
    # Display results
    display_results(results)

if __name__ == "__main__":
    main() 