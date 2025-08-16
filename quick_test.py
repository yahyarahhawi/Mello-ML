#!/usr/bin/env python3
"""
Quick validation that synthetic generation is working correctly.
"""

import logging
from profile_generator import ProfileGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_user():
    """Test generating a single synthetic user."""
    
    print("🧪 Testing single user generation...")
    
    # Initialize generator
    profile_generator = ProfileGenerator()
    
    # Generate one user
    print("🎭 Generating synthetic user...")
    user = profile_generator.generate_synthetic_user()
    
    if user:
        print(f"✅ Successfully generated: {user.name}")
        print(f"📊 Profile data: {user.profile_data.get('major', 'Unknown major')}")
        print(f"📚 Books: {len(user.metadata.get('frontend_data', {}).get('books', {}).get('favoriteBooks', []))} books")
        print(f"🎬 Movies: {len(user.metadata.get('frontend_data', {}).get('movies', {}).get('favoriteMovies', []))} movies")
        print(f"🎵 Artists: {len(user.metadata.get('frontend_data', {}).get('music', {}).get('musicArtists', []))} artists")
        print(f"🧠 Personality traits: {len(user.personality_responses)} traits")
        
        # Test dual profile generation
        print("\n📖 Testing dual profile generation...")
        profile_success = profile_generator.generate_dual_profiles(user)
        
        if profile_success:
            print(f"✅ Generated interests profile: {len(user.interests_profile)} chars")
            print(f"✅ Generated personality profiles: {len(user.personality_profile)} traits")
            
            # Show preview
            print(f"\n📄 Interests preview: {user.interests_profile[:100]}...")
            print(f"🧠 Personality traits: {list(user.personality_profile.keys())}")
            
            return True
        else:
            print("❌ Failed to generate dual profiles")
            return False
    else:
        print("❌ Failed to generate synthetic user")
        return False

if __name__ == "__main__":
    success = test_single_user()
    if success:
        print("\n🎉 Single user generation is working correctly!")
        print("💡 Ready for full population generation!")
    else:
        print("\n❌ Test failed!")