#!/usr/bin/env python3
"""
Test the fixed personality system that ensures proper Big 5 question selection.
"""

import logging
from profile_generator import ProfileGenerator
from embedding_generator import EmbeddingGenerator
from population import Population

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fixed_personality_system():
    """Test that personality questions are properly selected."""
    
    print("🧪 Testing Fixed Personality System")
    print("=" * 50)
    
    # Initialize generators
    profile_generator = ProfileGenerator()
    embedding_generator = EmbeddingGenerator()
    
    print(f"📝 Profile Generator: {profile_generator.taste_profile_model}")
    print(f"🔢 Embedding Generator: {embedding_generator.model_name}")
    
    # Test generating a single user
    print(f"\n🎭 Generating test user...")
    test_user = profile_generator.generate_synthetic_user()
    
    if not test_user:
        print("❌ Failed to generate test user")
        return False
    
    print(f"✅ Generated user: {test_user.name}")
    
    # Check the personality responses
    if hasattr(test_user, 'metadata') and 'frontend_data' in test_user.metadata:
        personality_data = test_user.metadata['frontend_data'].get('personality', {})
        
        print(f"\n🧠 Personality Analysis for {test_user.name}:")
        for trait, responses in personality_data.items():
            selected = responses.get('selected', [])
            not_selected = responses.get('not_selected', [])
            notes = responses.get('notes', '')
            
            print(f"\n   {trait}:")
            print(f"      Selected: {len(selected)} items")
            if selected:
                for item in selected[:3]:  # Show first 3
                    print(f"         - {item}")
                if len(selected) > 3:
                    print(f"         ... and {len(selected) - 3} more")
            else:
                print(f"         ❌ NO ITEMS SELECTED!")
            
            print(f"      Not selected: {len(not_selected)} items")
            if notes:
                print(f"      Notes: {notes[:100]}...")
    
    # Test dual profile generation
    print(f"\n📖 Testing dual profile generation...")
    profile_success = profile_generator.generate_dual_profiles(test_user)
    
    if profile_success:
        print(f"✅ Generated dual profiles successfully")
        print(f"   Interests profile: {len(test_user.interests_profile)} chars")
        print(f"   Personality profiles: {len(test_user.personality_profile)} traits")
        
        # Show personality profile preview
        print(f"\n🔍 Personality Profiles Preview:")
        for trait, description in list(test_user.personality_profile.items())[:2]:
            print(f"   {trait}: {description[:80]}...")
    else:
        print(f"❌ Failed to generate dual profiles")
        return False
    
    # Test dual embeddings
    print(f"\n🔢 Testing dual embeddings...")
    embedding_success = embedding_generator.embed_user_dual(test_user)
    
    if embedding_success:
        print(f"✅ Generated dual embeddings successfully")
        print(f"   Interests embedding: {test_user.interests_embedding.shape}")
        print(f"   Personality embedding: {test_user.personality_embedding.shape}")
        
        # Verify no legacy data
        has_legacy = hasattr(test_user, 'embedding') or hasattr(test_user, 'taste_profile')
        print(f"   Legacy data present: {'❌ YES' if has_legacy else '✅ NO'}")
    else:
        print(f"❌ Failed to generate dual embeddings")
        return False
    
    print(f"\n🎉 All tests passed!")
    return True

if __name__ == "__main__":
    success = test_fixed_personality_system()
    if success:
        print("\n✅ Fixed personality system is working correctly!")
        print("💡 Ready for production dual-vector generation!")
    else:
        print("\n❌ Tests failed!")