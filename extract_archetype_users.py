import json
import numpy as np
import argparse
from collections import Counter
import re

def load_book_embeddings_200():
    """Load the book embeddings from the 200-person dataset"""
    try:
        # Try the randomized version first, then fall back to original
        filenames = [
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

def get_all_archetypes(profiles):
    """Extract all unique archetypes from the profiles"""
    archetypes = []
    for profile in profiles:
        if 'personality_archetype' in profile:
            archetypes.append(profile['personality_archetype'])
    
    return list(set(archetypes))

def find_users_by_archetype(profiles, target_archetype, exact_match=True):
    """Find users with a specific archetype"""
    matching_users = []
    
    for profile in profiles:
        if 'personality_archetype' in profile:
            profile_archetype = profile['personality_archetype']
            
            if exact_match:
                # Exact match
                if profile_archetype.lower() == target_archetype.lower():
                    matching_users.append(profile)
            else:
                # Partial match (case-insensitive)
                if target_archetype.lower() in profile_archetype.lower():
                    matching_users.append(profile)
    
    return matching_users

def analyze_archetype_distribution(profiles):
    """Analyze the distribution of archetypes in the dataset"""
    archetype_counts = Counter()
    
    for profile in profiles:
        if 'personality_archetype' in profile:
            archetype = profile['personality_archetype']
            archetype_counts[archetype] += 1
    
    print(f"\n{'='*80}")
    print("ARCHETYPE DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    print(f"Total profiles analyzed: {len(profiles)}")
    print(f"Profiles with archetype labels: {sum(archetype_counts.values())}")
    print(f"Unique archetypes found: {len(archetype_counts)}")
    
    print(f"\nArchetype Distribution:")
    print(f"{'Count':<6} {'Archetype':<60}")
    print("-" * 66)
    
    for archetype, count in archetype_counts.most_common():
        percentage = (count / len(profiles)) * 100
        print(f"{count:<6} {archetype:<60}")
        print(f"       ({percentage:.1f}% of total)")
    
    return archetype_counts

def extract_users_by_archetype(profiles, target_archetype, exact_match=True):
    """Extract and display users with a specific archetype"""
    matching_users = find_users_by_archetype(profiles, target_archetype, exact_match)
    
    if not matching_users:
        print(f"\nNo users found with archetype: '{target_archetype}'")
        if exact_match:
            print("Try using --partial-match for partial matching")
        return []
    
    print(f"\n{'='*80}")
    print(f"USERS WITH ARCHETYPE: {target_archetype.upper()}")
    print(f"{'='*80}")
    print(f"Found {len(matching_users)} matching users")
    print(f"Match type: {'Exact' if exact_match else 'Partial'}")
    
    print(f"\nMatching Users:")
    print(f"{'Rank':<4} {'User Name':<25} {'Archetype':<50}")
    print("-" * 79)
    
    for i, profile in enumerate(matching_users, 1):
        name = profile.get('name', 'Unknown')
        archetype = profile.get('personality_archetype', 'Unknown')
        print(f"{i:<4} {name:<25} {archetype:<50}")
    
    # Show some statistics about these users
    print(f"\nStatistics:")
    print(f"- Total matching users: {len(matching_users)}")
    print(f"- Percentage of total dataset: {(len(matching_users) / len(profiles)) * 100:.1f}%")
    
    return matching_users

def search_archetypes_by_keyword(profiles, keyword):
    """Search for archetypes containing a specific keyword"""
    matching_archetypes = []
    
    for profile in profiles:
        if 'personality_archetype' in profile:
            archetype = profile['personality_archetype']
            if keyword.lower() in archetype.lower():
                if archetype not in matching_archetypes:
                    matching_archetypes.append(archetype)
    
    return matching_archetypes

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Extract users based on personality archetypes from the dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_archetype_users.py
  python extract_archetype_users.py "LGBTQ+ reader exploring queer literature and identity stories"
  python extract_archetype_users.py "LGBTQ+" --partial-match
  python extract_archetype_users.py --list-archetypes
  python extract_archetype_users.py --search "queer"
        """
    )
    
    parser.add_argument('archetype', nargs='?', 
                       default='LGBTQ+ reader exploring queer literature and identity stories',
                       help='Archetype to search for (default: LGBTQ+ reader)')
    parser.add_argument('--partial-match', action='store_true',
                       help='Use partial matching instead of exact matching')
    parser.add_argument('--list-archetypes', action='store_true',
                       help='List all available archetypes in the dataset')
    parser.add_argument('--search', type=str,
                       help='Search for archetypes containing a keyword')
    parser.add_argument('--distribution', action='store_true',
                       help='Show archetype distribution analysis')
    
    args = parser.parse_args()
    
    print("Loading book embeddings from 200-person dataset...")
    profiles = load_book_embeddings_200()
    if not profiles:
        return
    
    if args.list_archetypes:
        # List all available archetypes
        archetypes = get_all_archetypes(profiles)
        print(f"\n{'='*80}")
        print("ALL AVAILABLE ARCHETYPES")
        print(f"{'='*80}")
        for i, archetype in enumerate(sorted(archetypes), 1):
            print(f"{i:2d}. {archetype}")
        return
    
    if args.search:
        # Search for archetypes containing keyword
        matching_archetypes = search_archetypes_by_keyword(profiles, args.search)
        print(f"\n{'='*80}")
        print(f"ARCHETYPES CONTAINING: '{args.search}'")
        print(f"{'='*80}")
        if matching_archetypes:
            for i, archetype in enumerate(matching_archetypes, 1):
                print(f"{i}. {archetype}")
        else:
            print(f"No archetypes found containing '{args.search}'")
        return
    
    if args.distribution:
        # Show archetype distribution
        analyze_archetype_distribution(profiles)
        return
    
    # Extract users by archetype
    exact_match = not args.partial_match
    extract_users_by_archetype(profiles, args.archetype, exact_match)

if __name__ == "__main__":
    main() 