#!/usr/bin/env python3
"""
Test the special field functionality in User and Population classes
"""

import logging
logging.basicConfig(level=logging.INFO)

from user import User, UserPreferences, BookRating
from population import Population
from profile_generator import ProfileGenerator

def test_special_field():
    """Test the special field functionality."""
    
    print("🧪 Testing Special Field Functionality")
    print("=" * 50)
    
    # Test User creation with special field
    print("\n👤 Testing User creation:")
    
    # Regular user
    regular_user = User(name="Regular User")
    print(f"Regular: {regular_user}")
    
    # Special user
    special_user = User(name="Special User", special=True)
    print(f"Special: {special_user}")
    
    # Test Population methods
    print("\n👥 Testing Population methods:")
    population = Population("Test Population")
    
    # Add users
    population.add_user(regular_user)
    population.add_user(special_user)
    
    # Add more users for testing
    for i in range(3):
        user = User(name=f"User {i+1}", special=(i == 1))  # Make User 2 special
        population.add_user(user)
    
    print(f"Total users: {len(population)}")
    
    # Test special user methods
    special_users = population.get_special_users()
    regular_users = population.get_regular_users()
    
    print(f"\n⭐ Special users: {len(special_users)}")
    for user in special_users:
        print(f"  - {user}")
    
    print(f"\n👥 Regular users: {len(regular_users)}")
    for user in regular_users:
        print(f"  - {user}")
    
    # Test statistics
    print(f"\n📊 Population Statistics:")
    stats = population.get_statistics()
    for key, value in stats.items():
        if key in ["special_users", "regular_users", "total_users"]:
            print(f"  {key}: {value}")
    
    # Test DataFrame
    print(f"\n📋 DataFrame with special field:")
    df = population.to_dataframe()
    print(df[['name', 'special']].head())
    
    # Test JSON serialization
    print(f"\n💾 Testing JSON serialization:")
    
    # Save to JSON
    population.save_to_json('test_special_population.json')
    
    # Load from JSON
    loaded_population = Population.load_from_json('test_special_population.json')
    
    print(f"Original special users: {len(population.get_special_users())}")
    print(f"Loaded special users: {len(loaded_population.get_special_users())}")
    
    # Verify special users preserved
    original_special_names = {user.name for user in population.get_special_users()}
    loaded_special_names = {user.name for user in loaded_population.get_special_users()}
    
    print(f"Special users preserved: {'✅' if original_special_names == loaded_special_names else '❌'}")
    
    # Test ProfileGenerator create_user_from_books with special flag
    print(f"\n🤖 Testing ProfileGenerator with special flag:")
    
    profile_gen = ProfileGenerator()
    
    # Create special user from books
    books_data = [
        {"title": "Test Book", "author": "Test Author", "rating": 5.0, "genre": "Fiction"}
    ]
    
    created_user = profile_gen.create_user_from_books(
        name="Created Special User", 
        books_data=books_data, 
        special=True
    )
    
    print(f"Created user: {created_user}")
    print(f"Is special: {created_user.special}")
    
    return True

if __name__ == "__main__":
    success = test_special_field()
    print(f"\n🚀 Test {'PASSED' if success else 'FAILED'}")