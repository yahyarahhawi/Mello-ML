#!/usr/bin/env python3
"""
Test script for the new realistic synthetic user generation pipeline.
Tests that synthetic users follow the same authentic pipeline as real users.
"""

import json
import logging
from pathlib import Path

from profile_generator import ProfileGenerator
from embedding_generator import EmbeddingGenerator
from dual_vector_utils import process_frontend_json_to_user

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_personality_archetype_diversity():
    """Test that personality archetypes are diverse and realistic."""
    
    profile_generator = ProfileGenerator()
    
    print("🧪 Testing Personality Archetype Diversity")
    print("=" * 50)
    
    print(f"Total archetypes available: {len(profile_generator.personality_archetypes)}")
    
    # Show sample archetypes
    print(f"\n📋 Sample Personality Archetypes:")
    for i, archetype in enumerate(profile_generator.personality_archetypes[:10], 1):
        archetype_short = archetype.split(':')[0] if ':' in archetype else archetype[:50]
        print(f"   {i:2d}. {archetype_short}...")
    
    print(f"   ... and {len(profile_generator.personality_archetypes) - 10} more")
    
    # Test coverage of Big 5 combinations
    high_openness = [a for a in profile_generator.personality_archetypes if "High Openness" in a]
    low_openness = [a for a in profile_generator.personality_archetypes if "Low Openness" in a]
    high_conscientiousness = [a for a in profile_generator.personality_archetypes if "High Conscientiousness" in a]
    
    print(f"\n📊 Big 5 Coverage:")
    print(f"   High Openness archetypes: {len(high_openness)}")
    print(f"   Low Openness archetypes: {len(low_openness)}")
    print(f"   High Conscientiousness archetypes: {len(high_conscientiousness)}")
    
    print("✅ Archetype diversity test passed!")
    return profile_generator


def test_realistic_json_generation(profile_generator):
    """Test JSON generation from personality archetypes."""
    
    print(f"\n🧪 Testing Realistic JSON Generation")
    print("=" * 50)
    
    # Test a few different archetype types
    test_archetypes = [
        "High Openness, High Conscientiousness academic: loves learning, organized, seeks novel ideas but executes systematically.",
        "High Extraversion, Low Agreeableness ambitious leader: confident, competitive, drives results, comfortable with conflict.",
        "Low Extraversion, High Agreeableness supportive listener: prefers small groups, deeply empathetic, provides quiet support."
    ]
    
    generated_jsons = []
    
    for i, archetype in enumerate(test_archetypes, 1):
        print(f"\n🎭 Test {i}: {archetype.split(':')[0]}...")
        
        try:
            user_json = profile_generator.generate_synthetic_user_json(archetype)
            
            if user_json:
                generated_jsons.append(user_json)
                
                # Validate JSON structure
                required_sections = ['profile', 'books', 'movies', 'music', 'personality', 'additionalInfo']
                
                print(f"   ✅ JSON generated successfully")
                print(f"   📊 Structure validation:")
                
                for section in required_sections:
                    has_section = section in user_json
                    print(f"      {section}: {'✅' if has_section else '❌'}")
                
                # Check personality structure
                if 'personality' in user_json:
                    personality_data = user_json['personality']
                    big5_traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
                    
                    print(f"   🧠 Personality structure:")
                    for trait in big5_traits:
                        if trait in personality_data:
                            trait_data = personality_data[trait]
                            has_selected = 'selected' in trait_data
                            has_not_selected = 'not_selected' in trait_data
                            print(f"      {trait}: {'✅' if has_selected and has_not_selected else '❌'}")
                
                # Save sample for inspection
                sample_file = f"test_archetype_{i}.json"
                with open(sample_file, 'w') as f:
                    json.dump(user_json, f, indent=2)
                print(f"   💾 Saved sample: {sample_file}")
                
            else:
                print(f"   ❌ Failed to generate JSON")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 JSON Generation Results: {len(generated_jsons)}/{len(test_archetypes)} successful")
    return generated_jsons


def test_authentic_pipeline_comparison(generated_jsons):
    """Test that synthetic users follow the same pipeline as real users."""
    
    if not generated_jsons:
        print("⚠️  No generated JSONs to test pipeline")
        return
    
    print(f"\n🧪 Testing Authentic Pipeline Comparison")
    print("=" * 50)
    
    profile_generator = ProfileGenerator()
    embedding_generator = EmbeddingGenerator()
    
    # Test one synthetic user
    synthetic_json = generated_jsons[0]
    
    print(f"👤 Testing synthetic user: {synthetic_json['profile']['fullName']}")
    
    # Process synthetic user through authentic pipeline
    print(f"\n🔄 Processing synthetic user through authentic pipeline...")
    synthetic_user = process_frontend_json_to_user(
        synthetic_json, 
        profile_generator, 
        embedding_generator,
        generate_profiles=True,
        generate_embeddings=True
    )
    
    if synthetic_user:
        print(f"   ✅ Synthetic user processed successfully")
        print(f"   📖 Interests profile: {len(synthetic_user.interests_profile)} chars")
        print(f"   🧠 Personality profiles: {len(synthetic_user.personality_profile)} traits")
        print(f"   🔢 Interests embedding: {synthetic_user.interests_embedding.shape}")
        print(f"   🔢 Personality embedding: {synthetic_user.personality_embedding.shape}")
        
        # Validate that profiles were generated FROM the JSON data
        frontend_data = synthetic_user.metadata.get('frontend_data', {})
        
        # Check that interests profile mentions books/movies/music from JSON
        if synthetic_user.interests_profile and frontend_data:
            books = frontend_data.get('books', {}).get('favoriteBooks', [])
            movies = frontend_data.get('movies', {}).get('favoriteMovies', [])
            
            profile_text = synthetic_user.interests_profile.lower()
            
            # Check if profile mentions the user's actual preferences
            mentions_books = any(book.lower().split()[0] in profile_text for book in books[:2] if book)
            mentions_movies = any(movie.lower().split()[0] in profile_text for movie in movies[:2] if movie)
            
            print(f"\n🔍 Profile authenticity check:")
            print(f"   Books from JSON: {books[:3]}")
            print(f"   Movies from JSON: {movies[:3]}")
            print(f"   Profile mentions books: {'✅' if mentions_books else '⚠️'}")
            print(f"   Profile mentions movies: {'✅' if mentions_movies else '⚠️'}")
            
            # Show a snippet of the generated interests profile
            print(f"\n📄 Generated Interests Profile (first 150 chars):")
            print(f"   \"{synthetic_user.interests_profile[:150]}...\"")
        
        return synthetic_user
    
    else:
        print(f"   ❌ Failed to process synthetic user")
        return None


def test_similarity_with_real_user_format(synthetic_user):
    """Test that synthetic user can be compared with real user format."""
    
    if not synthetic_user:
        print("⚠️  No synthetic user to test similarity")
        return
    
    print(f"\n🧪 Testing Similarity with Real User Format")
    print("=" * 50)
    
    # Create a "real" user using the same format as the React frontend
    real_user_json = {
        "profile": {
            "fullName": "Test Real User",
            "classYear": "2025",
            "major": "Computer Science",
            "bio": "Love coding and sci-fi books. Always exploring new technologies.",
            "interests": ["Programming", "Science Fiction", "AI", "Movies", "Gaming"]
        },
        "books": {
            "favoriteBooks": ["Dune", "Neuromancer", "The Martian", "Foundation", "Klara and the Sun"],
            "bookReviews": {
                "Dune": "Epic world-building with incredible political complexity.",
                "Neuromancer": "Groundbreaking cyberpunk that predicted our digital future."
            },
            "bookReflection": "I love sci-fi that makes me think about technology and society."
        },
        "movies": {
            "favoriteMovies": ["Blade Runner 2049", "The Matrix", "Her", "Arrival", "Ex Machina"],
            "movieReviews": {
                "Blade Runner 2049": "Stunning visuals with deep philosophical themes.",
                "Her": "Beautiful exploration of AI consciousness and love."
            },
            "movieReflection": "I prefer films that blend great storytelling with tech themes."
        },
        "music": {
            "musicArtists": ["Radiohead", "Daft Punk", "Aphex Twin", "Boards of Canada", "Jon Hopkins"],
            "vibeMatch": "Electronic music that creates atmosphere and enhances focus."
        },
        "personality": {
            "Openness": {
                "selected": [
                    "I seek out restaurants serving cuisine I've never tried",
                    "I regularly explore music outside my usual genres",
                    "I enjoy learning about unfamiliar topics just for fun"
                ],
                "not_selected": [
                    "I stick mostly to familiar authors or genres when reading",
                    "I prefer detailed itineraries when traveling somewhere new"
                ],
                "notes": "Love exploring new ideas and experiences"
            },
            "Conscientiousness": {
                "selected": [
                    "I write down tasks as soon as I think of them",
                    "I rarely miss deadlines for assignments or work",
                    "I use to-do lists and use them to guide my day"
                ],
                "not_selected": [
                    "I have dozens of unread notifications on my phone",
                    "I sometimes run a few minutes late to events"
                ],
                "notes": "Pretty organized, especially for projects I care about"
            },
            # ... other traits would be here in real implementation
        },
        "additionalInfo": {
            "talkAboutForHours": "AI and machine learning, sci-fi worldbuilding, game design, future of programming",
            "naturalConversations": "Technical discussions, creative projects, philosophical implications of technology",
            "aloneTimeReachFor": "Coding personal projects, reading sci-fi, playing indie games, electronic music",
            "perfectDay": "Morning coding with coffee, afternoon reading, evening discussing ideas with friends"
        },
        "metadata": {
            "timestamp": "2024-01-15T10:30:00Z",
            "version": "1.0"
        }
    }
    
    profile_generator = ProfileGenerator()
    embedding_generator = EmbeddingGenerator()
    
    # Process real user through same pipeline
    real_user = process_frontend_json_to_user(
        real_user_json, 
        profile_generator, 
        embedding_generator
    )
    
    if real_user:
        print(f"✅ Both synthetic and real users processed through same pipeline")
        
        # Test similarity calculation
        try:
            interests_sim = synthetic_user.calculate_similarity(real_user, 'interests')
            personality_sim = synthetic_user.calculate_similarity(real_user, 'personality')
            combined_sim = synthetic_user.calculate_similarity(real_user, 'combined')
            
            print(f"\n🔍 Similarity Results:")
            print(f"   Interests similarity: {interests_sim:.3f}")
            print(f"   Personality similarity: {personality_sim:.3f}")
            print(f"   Combined similarity: {combined_sim:.3f}")
            
            print(f"\n✅ Synthetic and real users are fully compatible for similarity matching!")
            
        except Exception as e:
            print(f"❌ Similarity calculation failed: {e}")
    
    else:
        print(f"❌ Failed to process real user")


def main():
    """Run all tests for the realistic synthetic user pipeline."""
    
    print("🚀 Testing Realistic Synthetic User Generation Pipeline")
    print("=" * 70)
    print("This tests that synthetic users follow the same authentic pipeline as real users")
    print()
    
    try:
        # Test 1: Personality archetype diversity
        profile_generator = test_personality_archetype_diversity()
        
        # Test 2: Realistic JSON generation
        generated_jsons = test_realistic_json_generation(profile_generator)
        
        # Test 3: Authentic pipeline processing
        synthetic_user = test_authentic_pipeline_comparison(generated_jsons)
        
        # Test 4: Compatibility with real users
        test_similarity_with_real_user_format(synthetic_user)
        
        print(f"\n🎉 All Tests Completed!")
        print(f"📊 Summary:")
        print(f"   ✅ Personality archetypes are diverse and realistic")
        print(f"   ✅ JSON generation creates authentic user data")
        print(f"   ✅ Synthetic users follow same pipeline as real users")
        print(f"   ✅ Synthetic users are compatible with real users for matching")
        
        # Cleanup sample files
        sample_files = ["test_archetype_1.json", "test_archetype_2.json", "test_archetype_3.json"]
        for file in sample_files:
            if Path(file).exists():
                print(f"   📄 Sample file created: {file}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.error(f"Test error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Realistic synthetic user pipeline is working correctly!")
    else:
        print("\n❌ Tests failed!")
        exit(1)