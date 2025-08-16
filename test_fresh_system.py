#!/usr/bin/env python3
"""
Test the complete fresh system with real user JSON and synthetic users.
"""

import logging
import numpy as np
from user import User
from profile_generator import ProfileGenerator
from embedding_generator import EmbeddingGenerator
from population import Population
from visualizer import Visualizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def test_complete_pipeline():
    """Test the complete fresh pipeline."""
    
    print("🧪 Testing Fresh Unified Personality System")
    print("=" * 50)
    
    # Initialize components
    print("\n1. Initializing components...")
    profile_gen = ProfileGenerator()
    embedding_gen = EmbeddingGenerator()
    population = Population("Test Population")
    visualizer = Visualizer()
    
    print(f"✅ Components initialized")
    print(f"   ProfileGenerator: {profile_gen.model} with {len(profile_gen.archetypes)} archetypes")
    print(f"   EmbeddingGenerator: {embedding_gen}")
    
    # Test loading real user
    print("\n2. Loading real user from JSON...")
    try:
        real_user_path = "personality-json/mello-profile-yahya-rahhawi-2025-08-15 (1).json"
        real_user = User.from_json_file(real_user_path)
        real_user.special = True  # Mark as special for visualization
        print(f"✅ Loaded real user: {real_user.name}")
        
        # Generate profiles for real user
        print(f"   Generating profiles for {real_user.name}...")
        profile_success = profile_gen.generate_complete_profiles(real_user)
        
        if profile_success:
            print(f"   ✅ Generated profiles for {real_user.name}")
            
            # Generate embeddings for real user
            print(f"   Generating embeddings for {real_user.name}...")
            embedding_success = embedding_gen.embed_user_complete(real_user)
            
            if embedding_success:
                print(f"   ✅ Generated embeddings for {real_user.name}")
                population.add_user(real_user)
            else:
                print(f"   ❌ Failed to generate embeddings for {real_user.name}")
        else:
            print(f"   ❌ Failed to generate profiles for {real_user.name}")
    
    except Exception as e:
        print(f"❌ Failed to load real user: {e}")
        print("   Continuing with synthetic users only...")
    
    # Test synthetic user generation
    print("\n3. Generating synthetic users...")
    
    synthetic_count = 5
    successful_synthetic = 0
    
    for i in range(synthetic_count):
        print(f"   Generating synthetic user {i+1}/{synthetic_count}...")
        
        # Generate synthetic user data
        user_data = profile_gen.generate_synthetic_user_data()
        
        if user_data:
            user = User.from_json_data(user_data)
            
            # Generate profiles
            if profile_gen.generate_complete_profiles(user):
                # Generate embeddings
                if embedding_gen.embed_user_complete(user):
                    population.add_user(user)
                    successful_synthetic += 1
                    print(f"      ✅ Complete success for {user.name}")
                else:
                    print(f"      ❌ Failed embeddings for {user.name}")
            else:
                print(f"      ❌ Failed profiles for {user.name}")
        else:
            print(f"      ❌ Failed to generate user data")
    
    print(f"\n✅ Generated {successful_synthetic}/{synthetic_count} synthetic users")
    
    # Population statistics
    print(f"\n4. Population Statistics:")
    stats = population.get_statistics()
    print(f"   Total users: {stats['total_users']}")
    print(f"   Users with profiles: {stats['users_with_profiles']}")
    print(f"   Users with embeddings: {stats['users_with_embeddings']}")
    
    if stats['embedding_stats']:
        print(f"   Embedding dimensions:")
        print(f"     Interests: {stats['embedding_stats'].get('interests_dims', 'N/A')}")
        print(f"     Combined: {stats['embedding_stats'].get('combined_dims', 'N/A')}")
    
    # Test similarity search
    if len(population.get_users_with_embeddings()) >= 2:
        print(f"\n5. Testing similarity search...")
        
        # Find a user to test with
        test_user = None
        for user in population.users:
            if user.get_combined_embedding() is not None:
                test_user = user
                break
        
        if test_user:
            similar_users = population.find_similar_users(test_user, mode='combined', top_k=3)
            print(f"   Most similar users to {test_user.name}:")
            
            for i, (similar_user, score) in enumerate(similar_users, 1):
                print(f"     {i}. {similar_user.name}: {score:.3f}")
        
        # Test trait-specific similarity
        print(f"\n   Testing trait-specific similarities for {test_user.name}:")
        traits = ['interests', 'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        
        for trait in traits:
            try:
                similar = population.find_similar_users(test_user, mode=trait, top_k=1)
                if similar:
                    most_similar, score = similar[0]
                    print(f"     {trait}: {most_similar.name} ({score:.3f})")
            except Exception as e:
                print(f"     {trait}: Error - {e}")
    
    # Save population
    print(f"\n6. Saving population...")
    population.save_to_json("test_population.json")
    print(f"   ✅ Saved to test_population.json")
    
    # Test visualization (if enough users)
    if len(population.get_users_with_embeddings()) >= 3:
        print(f"\n7. Testing visualization...")
        
        try:
            # Test PCA
            fig_pca = visualizer.plot_population_pca(population, mode='combined', save_path='test_pca.png')
            print(f"   ✅ Created PCA visualization (saved to test_pca.png)")
            
            # Test UMAP (if enough users)
            if len(population.get_users_with_embeddings()) >= 10:
                fig_umap = visualizer.plot_population_umap(population, mode='combined', save_path='test_umap.png')
                print(f"   ✅ Created UMAP visualization (saved to test_umap.png)")
            else:
                print(f"   ⚠️  Need 10+ users for UMAP (have {len(population.get_users_with_embeddings())})")
            
            # Close figures to free memory
            import matplotlib.pyplot as plt
            plt.close('all')
            
        except Exception as e:
            print(f"   ❌ Visualization failed: {e}")
            print(f"      (This might be due to missing dependencies: sklearn, umap-learn, matplotlib)")
    
    print(f"\n🎉 Fresh system test completed!")
    print(f"   Population: {len(population)} total users")
    print(f"   Architecture: 768D interests + 5×768D traits = 4608D combined")
    print(f"   Approach: Unified personality profiling (legacy-style)")
    
    return population

if __name__ == "__main__":
    population = test_complete_pipeline()