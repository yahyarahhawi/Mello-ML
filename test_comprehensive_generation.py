#!/usr/bin/env python3
"""
Test script for comprehensive synthetic user generation.
Generates a few users with full JSON structure and validates the output.
"""

import json
import logging
from pathlib import Path

from profile_generator import ProfileGenerator
from embedding_generator import EmbeddingGenerator
from user import User

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_comprehensive_user_generation():
    """Test comprehensive JSON-based user generation."""
    
    # Initialize generator
    profile_generator = ProfileGenerator()
    
    # Test personality types
    test_personalities = [
        "Academic philosopher who loves ancient wisdom and existential questions",
        "Tech entrepreneur fascinated by science fiction and innovation",
        "Environmental activist drawn to nature writing and climate literature"
    ]
    
    generated_users = []
    
    for i, personality_type in enumerate(test_personalities, 1):
        print(f"\n🧪 Test {i}: Generating user with personality: {personality_type[:50]}...")
        
        try:
            user = profile_generator.generate_synthetic_user(personality_type)
            
            if user:
                generated_users.append(user)
                print(f"✅ Successfully generated: {user.name}")
                
                # Validate structure
                print(f"   📊 Validation:")
                print(f"      Has frontend_data: {bool(user.metadata.get('frontend_data'))}")
                
                if user.metadata.get('frontend_data'):
                    frontend_data = user.metadata['frontend_data']
                    
                    # Check main sections
                    sections = ['profile', 'books', 'movies', 'music', 'personality', 'additionalInfo']
                    for section in sections:
                        has_section = section in frontend_data
                        print(f"      Has {section}: {has_section}")
                    
                    # Check personality structure
                    if 'personality' in frontend_data:
                        personality_data = frontend_data['personality']
                        traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
                        for trait in traits:
                            has_trait = trait in personality_data
                            if has_trait:
                                has_selected = 'selected' in personality_data[trait]
                                has_not_selected = 'not_selected' in personality_data[trait]
                                print(f"         {trait}: {has_selected and has_not_selected}")
                
                # Save sample JSON for inspection
                sample_file = f"sample_generated_user_{i}.json"
                with open(sample_file, 'w') as f:
                    if user.metadata.get('frontend_data'):
                        json.dump(user.metadata['frontend_data'], f, indent=2)
                        print(f"   💾 Saved sample JSON: {sample_file}")
                
            else:
                print(f"❌ Failed to generate user for: {personality_type[:50]}...")
                
        except Exception as e:
            print(f"❌ Error generating user: {e}")
            logger.error(f"Generation error: {e}")
    
    return generated_users


def test_dual_vector_pipeline(users):
    """Test the complete dual-vector pipeline on generated users."""
    
    if not users:
        print("⚠️  No users to test dual-vector pipeline")
        return
    
    print(f"\n🚀 Testing dual-vector pipeline on {len(users)} users...")
    
    # Initialize generators
    profile_generator = ProfileGenerator()
    embedding_generator = EmbeddingGenerator()
    
    successful_users = []
    
    for user in users:
        print(f"\n👤 Processing {user.name}...")
        
        try:
            # Generate dual profiles
            print("   📖 Generating dual profiles...")
            profile_success = profile_generator.generate_dual_profiles(user)
            
            if profile_success:
                print("   ✅ Dual profiles generated successfully")
                print(f"      Interests profile: {len(user.interests_profile) if user.interests_profile else 0} chars")
                print(f"      Personality profiles: {len(user.personality_profile) if user.personality_profile else 0} traits")
                
                # Generate dual embeddings
                print("   🔢 Generating dual embeddings...")
                embedding_success = embedding_generator.embed_user_dual(user)
                
                if embedding_success:
                    print("   ✅ Dual embeddings generated successfully")
                    print(f"      Interests embedding: {user.interests_embedding.shape}")
                    print(f"      Personality embedding: {user.personality_embedding.shape}")
                    successful_users.append(user)
                else:
                    print("   ❌ Failed to generate dual embeddings")
            else:
                print("   ❌ Failed to generate dual profiles")
                
        except Exception as e:
            print(f"   ❌ Error processing {user.name}: {e}")
            logger.error(f"Pipeline error for {user.name}: {e}")
    
    print(f"\n📊 Pipeline Results:")
    print(f"   Successfully processed: {len(successful_users)}/{len(users)} users")
    
    if successful_users:
        print(f"   ✅ All processed users have:")
        print(f"      - Comprehensive JSON structure")
        print(f"      - Interests profiles + personality profiles")
        print(f"      - 3072D interests embeddings + 3840D personality embeddings")
        
        # Test similarity calculation
        if len(successful_users) >= 2:
            user1, user2 = successful_users[0], successful_users[1]
            print(f"\n🔍 Testing similarity calculation between {user1.name} and {user2.name}:")
            
            try:
                interests_sim = user1.calculate_similarity(user2, 'interests')
                personality_sim = user1.calculate_similarity(user2, 'personality')
                combined_sim = user1.calculate_similarity(user2, 'combined')
                
                print(f"   Interests similarity: {interests_sim:.3f}")
                print(f"   Personality similarity: {personality_sim:.3f}")
                print(f"   Combined similarity: {combined_sim:.3f}")
                
            except Exception as e:
                print(f"   ❌ Similarity calculation error: {e}")
    
    return successful_users


def main():
    """Run comprehensive tests."""
    print("🧪 Testing Comprehensive Synthetic User Generation")
    print("=" * 60)
    
    try:
        # Test user generation
        users = test_comprehensive_user_generation()
        
        # Test dual-vector pipeline
        if users:
            processed_users = test_dual_vector_pipeline(users)
            
            print(f"\n🎉 Test Summary:")
            print(f"   Generated users: {len(users)}")
            print(f"   Successfully processed: {len(processed_users) if users else 0}")
            print(f"   Success rate: {len(processed_users)/len(users)*100 if users else 0:.1f}%")
            
            # Cleanup sample files
            for i in range(1, 4):
                sample_file = f"sample_generated_user_{i}.json"
                if Path(sample_file).exists():
                    print(f"   📄 Sample JSON saved: {sample_file}")
        else:
            print(f"\n❌ No users were successfully generated")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.error(f"Main test error: {e}")


if __name__ == "__main__":
    main()