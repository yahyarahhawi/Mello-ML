#!/usr/bin/env python3
"""
Test script to verify the save functionality works with dual-vector users
"""

from user import User, UserPreferences, BookRating
from population import Population
import tempfile
import os

def test_dual_vector_save():
    """Test saving dual-vector users to JSON"""
    
    # Create a simple user with minimal data
    books = [BookRating(title="Test Book", author="Test Author", rating=5.0)]
    preferences = UserPreferences(books=books, movies=["Test Movie"], music=["Test Artist"])
    
    user = User(name="Test User", preferences=preferences)
    
    # Add dual-vector data (simulate what would come from the pipeline)
    user.interests_profile = "Test interests profile"
    user.personality_profile = {"Openness": "High openness description"}
    
    # Don't add legacy fields - this simulates a pure dual-vector user
    
    print("✅ Created dual-vector user")
    print(f"   Name: {user.name}")
    print(f"   Has interests_profile: {hasattr(user, 'interests_profile')}")
    print(f"   Has personality_profile: {hasattr(user, 'personality_profile')}")
    print(f"   Has taste_profile: {hasattr(user, 'taste_profile')}")
    print(f"   Has embedding: {hasattr(user, 'embedding')}")
    
    # Test to_dict conversion
    try:
        user_dict = user.to_dict()
        print("✅ to_dict() conversion successful")
        print(f"   taste_profile in dict: {user_dict.get('taste_profile')}")
        print(f"   embedding in dict: {user_dict.get('embedding')}")
    except Exception as e:
        print(f"❌ to_dict() failed: {e}")
        return False
    
    # Test population save
    population = Population(name="Test Population")
    population.add_user(user)
    
    # Use temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        population.save_to_json(tmp_path)
        print("✅ Population save successful")
        
        # Clean up
        os.unlink(tmp_path)
        return True
        
    except Exception as e:
        print(f"❌ Population save failed: {e}")
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return False

if __name__ == "__main__":
    print("🧪 Testing dual-vector user save functionality...")
    success = test_dual_vector_save()
    
    if success:
        print("\n🎉 All tests passed! The save functionality is working.")
    else:
        print("\n💥 Tests failed. Check the error messages above.")