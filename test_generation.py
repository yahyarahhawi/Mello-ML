#!/usr/bin/env python3
"""
Quick test script to verify synthetic data generation works correctly.
"""

import logging
from generate_synthetic_data import generate_synthetic_population, load_synthetic_population

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_generation():
    """Test generating a small synthetic population."""
    
    print("🧪 Testing synthetic data generation...")
    
    # Generate just 3 users for testing
    population = generate_synthetic_population(
        size=3,
        output_dir="test_synthetic_data"
    )
    
    if population and len(population) > 0:
        print(f"\n✅ Successfully generated {len(population)} users!")
        
        # Check that users have dual embeddings
        stats = population.get_statistics()
        print(f"📊 Population Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Test loading the data back
        print(f"\n📂 Testing data loading...")
        loaded_population = load_synthetic_population("test_synthetic_data/synthetic_population.json")
        
        if loaded_population and len(loaded_population) == len(population):
            print(f"✅ Successfully loaded {len(loaded_population)} users!")
            return True
        else:
            print("❌ Failed to load data correctly")
            return False
    else:
        print("❌ Failed to generate users")
        return False

if __name__ == "__main__":
    success = test_generation()
    if success:
        print("\n🎉 Synthetic data generation is working correctly!")
        print("💡 You can now run: python generate_synthetic_data.py --size 20")
    else:
        print("\n❌ Test failed!")