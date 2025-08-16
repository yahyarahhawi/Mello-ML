#!/usr/bin/env python3
"""
Quick test to verify population statistics work after attribute fixes.
"""

from population import Population
from user import User

def test_population_statistics():
    """Test that population statistics work without attribute errors."""
    
    print("🧪 Testing Population Statistics Fix")
    print("=" * 40)
    
    # Create a test population
    population = Population("Test Population")
    
    # Add a user with dual embeddings only
    user = User("Test User")
    user.interests_embedding = [1.0] * 3072  # Mock interests embedding
    user.personality_embedding = [0.5] * 3840  # Mock personality embedding
    user.interests_profile = "Test interests profile"
    user.personality_profile = {"Openness": "Test openness description"}
    
    population.add_user(user)
    
    # Test statistics
    try:
        stats = population.get_statistics()
        print("✅ Population statistics generated successfully!")
        print("📊 Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_population_statistics()
    if success:
        print("\n✅ Population fix works correctly!")
    else:
        print("\n❌ Population fix failed!")