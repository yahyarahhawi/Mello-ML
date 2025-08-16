#!/usr/bin/env python3
"""
Comprehensive test script for the dual-vector system workflow.
Tests all components working together end-to-end.
"""

import json
import logging
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Any

from user import User
from population import Population
from profile_generator import ProfileGenerator
from embedding_generator import EmbeddingGenerator
from dual_vector_utils import (
    process_frontend_json_to_user,
    find_best_matches,
    compare_similarity_modes,
    analyze_population_embeddings,
    validate_dual_vector_user,
    create_compatibility_report
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_users_data() -> List[Dict[str, Any]]:
    """Create test data for multiple users with different personality profiles."""
    
    base_template = {
        "metadata": {
            "timestamp": "2024-01-15T10:30:00Z",
            "version": "1.0"
        }
    }
    
    users_data = [
        {
            **base_template,
            "profile": {
                "fullName": "Alex Chen",
                "classYear": "2025",
                "major": "Computer Science",
                "bio": "Love sci-fi books and indie films. Always building something new.",
                "interests": ["Programming", "Science Fiction", "Film", "AI", "Philosophy"]
            },
            "books": {
                "favoriteBooks": ["Dune", "Neuromancer", "Foundation", "The Left Hand of Darkness"],
                "bookReviews": {
                    "Dune": "Epic world-building with complex political themes.",
                    "Neuromancer": "Groundbreaking cyberpunk that predicted our digital future."
                },
                "bookReflection": "I love science fiction that explores big ideas about technology and society."
            },
            "movies": {
                "favoriteMovies": ["Blade Runner 2049", "Her", "Ex Machina", "Arrival"],
                "movieReviews": {
                    "Blade Runner 2049": "Stunning visuals with deep themes about humanity.",
                    "Her": "Beautiful exploration of love and technology."
                },
                "movieReflection": "I prefer thoughtful sci-fi that makes me question reality."
            },
            "music": {
                "musicArtists": ["Radiohead", "Bon Iver", "Boards of Canada", "Brian Eno"],
                "vibeMatch": "Atmospheric, experimental electronic music."
            },
            "personality": {
                "Openness": {
                    "selected": ["seek-new-cuisine", "explore-music", "experimental-art", "learn-for-fun"],
                    "not_selected": ["familiar-reading", "routine-shopping", "rewatch-comfort"],
                    "notes": "Very open to new experiences."
                },
                "Conscientiousness": {
                    "selected": ["write-tasks-immediately", "rarely-miss-deadlines", "use-todo-lists"],
                    "not_selected": ["unread-notifications", "sometimes-late"],
                    "notes": "Quite organized and reliable."
                },
                "Extraversion": {
                    "selected": ["think-aloud", "invite-others", "energized-groups"],
                    "not_selected": ["eat-alone", "social-events-tire", "need-recharge"],
                    "notes": "More extraverted than average."
                },
                "Agreeableness": {
                    "selected": ["split-bills-evenly", "offer-help-proactively", "adjust-plans-others"],
                    "not_selected": ["push-for-choice", "own-needs-first"],
                    "notes": "Generally cooperative and helpful."
                },
                "Neuroticism": {
                    "selected": ["reread-emails", "make-backup-plans", "worry-embarrassing"],
                    "not_selected": ["dont-dwell-uncontrollable", "calm-social-events"],
                    "notes": "Some anxiety but generally stable."
                }
            },
            "additionalInfo": {
                "talkAboutForHours": "AI and the future of humanity, consciousness, sci-fi worlds",
                "naturalConversations": "Deep philosophical discussions, tech trends, creative projects",
                "aloneTimeReachFor": "Coding, reading, electronic music, thinking",
                "perfectDay": "Morning coding, afternoon with friends, evening reading sci-fi"
            }
        },
        {
            **base_template,
            "profile": {
                "fullName": "Maya Rodriguez",
                "classYear": "2026", 
                "major": "Environmental Studies",
                "bio": "Passionate about sustainability and social justice. Love hiking and poetry.",
                "interests": ["Environment", "Poetry", "Hiking", "Social Justice", "Meditation"]
            },
            "books": {
                "favoriteBooks": ["Braiding Sweetgrass", "The Overstory", "Silent Spring", "Parable of the Sower"],
                "bookReviews": {
                    "Braiding Sweetgrass": "Beautiful blend of indigenous wisdom and scientific knowledge.",
                    "The Overstory": "Transformative perspective on our relationship with trees."
                },
                "bookReflection": "I'm drawn to books about our connection to nature and environmental consciousness."
            },
            "movies": {
                "favoriteMovies": ["Princess Mononoke", "WALL-E", "The Tree of Life", "Nomadland"],
                "movieReviews": {
                    "Princess Mononoke": "Perfect balance of environmental themes and storytelling.",
                    "WALL-E": "Powerful environmental message wrapped in beautiful animation."
                },
                "movieReflection": "I love films that explore our relationship with nature and each other."
            },
            "music": {
                "musicArtists": ["Bon Iver", "Sufjan Stevens", "Joanna Newsom", "Fleet Foxes"],
                "vibeMatch": "Indie folk with natural, organic sounds and introspective lyrics."
            },
            "personality": {
                "Openness": {
                    "selected": ["seek-new-cuisine", "explore-neighborhoods", "learn-for-fun", "debate-perspectives"],
                    "not_selected": ["same-apps", "routine-shopping"],
                    "notes": "Very open, especially to new ideas about sustainability."
                },
                "Conscientiousness": {
                    "selected": ["make-bed-automatically", "tidy-up-soon", "track-expenses", "prepare-weeks-advance"],
                    "not_selected": ["unread-notifications", "lose-momentum-goals"],
                    "notes": "Disciplined about environmental practices."
                },
                "Extraversion": {
                    "selected": ["smaller-conversations", "quiet-study", "need-recharge"],
                    "not_selected": ["volunteer-speak", "talk-strangers", "energized-groups"],
                    "notes": "More introverted, prefer meaningful one-on-one connections."
                },
                "Agreeableness": {
                    "selected": ["split-bills-evenly", "others-choose-activity", "offer-help-proactively", "remember-birthdays"],
                    "not_selected": ["push-for-choice", "own-needs-first"],
                    "notes": "Very agreeable and empathetic."
                },
                "Neuroticism": {
                    "selected": ["lose-sleep-events", "replay-conversations", "second-guess-decisions"],
                    "not_selected": ["calm-social-events", "recover-quickly"],
                    "notes": "Sometimes anxious about environmental issues and social situations."
                }
            },
            "additionalInfo": {
                "talkAboutForHours": "Climate change solutions, indigenous wisdom, poetry, spiritual practices",
                "naturalConversations": "Environmental activism, personal growth, art, social justice",
                "aloneTimeReachFor": "Hiking, journaling, meditation, reading poetry",
                "perfectDay": "Sunrise hike, afternoon reading outside, evening with close friends"
            }
        },
        {
            **base_template,
            "profile": {
                "fullName": "Jordan Kim",
                "classYear": "2024",
                "major": "Psychology", 
                "bio": "Fascinated by human behavior and mental health. Love true crime and cooking.",
                "interests": ["Psychology", "True Crime", "Cooking", "Mental Health", "Podcasts"]
            },
            "books": {
                "favoriteBooks": ["The Body Keeps the Score", "Thinking, Fast and Slow", "In Cold Blood", "The Gifts of Imperfection"],
                "bookReviews": {
                    "The Body Keeps the Score": "Groundbreaking insights into trauma and healing.",
                    "Thinking, Fast and Slow": "Fascinating exploration of cognitive biases."
                },
                "bookReflection": "I'm drawn to books that help me understand why people think and act the way they do."
            },
            "movies": {
                "favoriteMovies": ["Mindhunter", "The Silence of the Lambs", "Good Will Hunting", "Inside Out"],
                "movieReviews": {
                    "Mindhunter": "Brilliant psychological analysis of criminal behavior.",
                    "Good Will Hunting": "Beautiful portrayal of therapy and personal growth."
                },
                "movieReflection": "I love psychological thrillers and films about human complexity."
            },
            "music": {
                "musicArtists": ["Phoebe Bridgers", "The National", "Lorde", "Taylor Swift"],
                "vibeMatch": "Emotionally intelligent indie and pop with introspective lyrics."
            },
            "personality": {
                "Openness": {
                    "selected": ["learn-for-fun", "debate-perspectives", "experiment-cooking"],
                    "not_selected": ["familiar-reading", "detailed-itineraries"],
                    "notes": "Open to new psychological theories and perspectives."
                },
                "Conscientiousness": {
                    "selected": ["write-tasks-immediately", "rarely-miss-deadlines", "prepare-weeks-advance", "charge-proactively"],
                    "not_selected": ["unread-notifications", "sometimes-late"],
                    "notes": "Very organized, especially for academic work."
                },
                "Extraversion": {
                    "selected": ["smaller-conversations", "think-aloud", "invite-others"],
                    "not_selected": ["eat-alone", "social-events-tire"],
                    "notes": "Balanced extroversion - social but selective."
                },
                "Agreeableness": {
                    "selected": ["offer-help-proactively", "check-before-ordering", "apologize-when-hurt", "adjust-plans-others"],
                    "not_selected": ["wait-to-trust", "own-needs-first"],
                    "notes": "Very empathetic and caring toward others."
                },
                "Neuroticism": {
                    "selected": ["reread-emails", "replay-conversations", "make-backup-plans"],
                    "not_selected": ["dont-dwell-uncontrollable", "laugh-off-mistakes"],
                    "notes": "Some anxiety but uses it productively for planning."
                }
            },
            "additionalInfo": {
                "talkAboutForHours": "Psychology theories, criminal psychology, mental health awareness, cooking techniques",
                "naturalConversations": "Deep personal discussions, psychological analysis, true crime theories",
                "aloneTimeReachFor": "Cooking, podcasts, psychology research, true crime documentaries",
                "perfectDay": "Morning cooking, afternoon studying psychology, evening discussing cases with friends"
            }
        }
    ]
    
    return users_data


def test_user_creation_and_validation():
    """Test user creation from frontend JSON and validation."""
    logger.info("🧪 Testing user creation and validation...")
    
    test_data = create_test_users_data()[0]  # Use first user for testing
    
    # Create user from JSON
    user = User.from_frontend_json(test_data)
    logger.info(f"✅ Created user: {user}")
    
    # Validate user structure
    validation = validate_dual_vector_user(user)
    logger.info(f"📊 Validation results: {validation}")
    
    # Check that basic data was loaded correctly
    assert user.name == "Alex Chen"
    assert user.profile_data['major'] == "Computer Science"
    assert len(user.personality_responses) == 5  # Big 5 traits
    
    logger.info("✅ User creation and validation tests passed!")
    return user


def test_profile_generation(user: User, profile_generator: ProfileGenerator):
    """Test interests and personality profile generation."""
    logger.info("🧪 Testing profile generation...")
    
    # Generate interests profile
    interests_success = profile_generator.generate_interests_profile(user)
    assert interests_success, "Failed to generate interests profile"
    assert user.interests_profile is not None
    logger.info(f"✅ Interests profile generated ({len(user.interests_profile)} chars)")
    
    # Generate personality profiles
    personality_success = profile_generator.generate_personality_profiles(user)
    assert personality_success, "Failed to generate personality profiles"
    assert user.personality_profile is not None
    assert len(user.personality_profile) == 5  # Big 5 traits
    logger.info(f"✅ Personality profiles generated for {len(user.personality_profile)} traits")
    
    # Test dual profile generation
    user2 = User.from_frontend_json(create_test_users_data()[1])  # Maya
    dual_success = profile_generator.generate_dual_profiles(user2)
    assert dual_success, "Failed to generate dual profiles"
    
    logger.info("✅ Profile generation tests passed!")
    return user


def test_embedding_generation(user: User, embedding_generator: EmbeddingGenerator):
    """Test interests and personality embedding generation."""
    logger.info("🧪 Testing embedding generation...")
    
    # Generate dual embeddings
    success = embedding_generator.embed_user_dual(user)
    assert success, "Failed to generate dual embeddings"
    
    # Validate embedding dimensions
    assert user.interests_embedding is not None
    assert user.personality_embedding is not None
    assert len(user.interests_embedding) == 3072, f"Expected 3072 dims, got {len(user.interests_embedding)}"
    assert len(user.personality_embedding) == 3840, f"Expected 3840 dims, got {len(user.personality_embedding)}"
    
    logger.info(f"✅ Dual embeddings generated successfully")
    logger.info(f"   Interests: {user.interests_embedding.shape}")
    logger.info(f"   Personality: {user.personality_embedding.shape}")
    
    logger.info("✅ Embedding generation tests passed!")
    return user


def test_similarity_calculations():
    """Test similarity calculations between users."""
    logger.info("🧪 Testing similarity calculations...")
    
    # Create test users
    profile_generator = ProfileGenerator()
    embedding_generator = EmbeddingGenerator()
    
    users_data = create_test_users_data()
    users = []
    
    for user_data in users_data:
        user = process_frontend_json_to_user(user_data, profile_generator, embedding_generator)
        users.append(user)
    
    logger.info(f"Created {len(users)} test users")
    
    # Test all similarity modes
    user1, user2 = users[0], users[1]  # Alex and Maya
    
    similarities = compare_similarity_modes(user1, user2)
    logger.info(f"Similarity between {user1.name} and {user2.name}:")
    for mode, sim in similarities.items():
        if sim is not None:
            logger.info(f"   {mode}: {sim:.3f}")
    
    # Test that similarities are in valid range
    for mode, sim in similarities.items():
        if sim is not None:
            assert -1 <= sim <= 1, f"Similarity {sim} out of range for mode {mode}"
    
    logger.info("✅ Similarity calculation tests passed!")
    return users


def test_population_operations(users: List[User]):
    """Test population-level operations with dual vectors."""
    logger.info("🧪 Testing population operations...")
    
    # Create population
    population = Population("Test Population")
    for user in users:
        population.add_user(user)
    
    logger.info(f"Created population with {len(population)} users")
    
    # Test different embedding modes
    modes = ['legacy', 'interests', 'personality', 'dual', 'combined']
    for mode in modes:
        try:
            users_with_embeddings = population.get_users_with_embeddings(mode)
            logger.info(f"   {mode}: {len(users_with_embeddings)} users")
            
            if len(users_with_embeddings) > 0:
                # Test embedding matrix
                if mode in ['interests', 'personality', 'combined']:
                    matrix = population.get_embedding_matrix(mode)
                    logger.info(f"   {mode} matrix shape: {matrix.shape}")
        except ValueError as e:
            logger.info(f"   {mode}: {e}")
    
    # Test similarity search
    target_user = users[0]  # Alex
    for mode in ['interests', 'personality', 'combined']:
        try:
            matches = population.find_similar_users(target_user, top_k=2, mode=mode)
            logger.info(f"Top {mode} matches for {target_user.name}:")
            for match_user, similarity in matches:
                logger.info(f"   {match_user.name}: {similarity:.3f}")
        except ValueError as e:
            logger.info(f"   {mode} similarity search: {e}")
    
    # Test detailed matches
    detailed_matches = find_best_matches(target_user, population, 'combined', 2)
    logger.info(f"Detailed matches for {target_user.name}:")
    for match_user, similarity, breakdown in detailed_matches:
        logger.info(f"   {match_user.name}: {similarity:.3f} (breakdown: {breakdown})")
    
    # Test population statistics
    stats = population.get_statistics()
    logger.info("Population statistics:")
    for key, value in stats.items():
        logger.info(f"   {key}: {value}")
    
    # Test population analysis
    analysis = analyze_population_embeddings(population)
    logger.info("Population analysis:")
    logger.info(f"   Coverage: {analysis['embedding_coverage']}")
    logger.info(f"   Dimensions: {analysis['embedding_dimensions']}")
    
    logger.info("✅ Population operation tests passed!")
    return population


def test_compatibility_reporting(users: List[User]):
    """Test compatibility reporting between users."""
    logger.info("🧪 Testing compatibility reporting...")
    
    user1, user2 = users[0], users[1]  # Alex and Maya
    
    # Create compatibility report
    report = create_compatibility_report(user1, user2)
    
    logger.info(f"Compatibility report for {user1.name} and {user2.name}:")
    logger.info(f"   Overall compatibility: {report['overall_compatibility']:.3f}")
    logger.info(f"   Similarities: {report['similarities']}")
    logger.info(f"   Shared books: {report['shared_interests']['books']}")
    logger.info(f"   Shared movies: {report['shared_interests']['movies']}")
    logger.info(f"   Shared music: {report['shared_interests']['music']}")
    
    assert report['overall_compatibility'] is not None
    assert isinstance(report['shared_interests'], dict)
    
    logger.info("✅ Compatibility reporting tests passed!")


def test_serialization_and_loading(user: User):
    """Test user serialization and loading with dual vectors."""
    logger.info("🧪 Testing serialization and loading...")
    
    # Save user to JSON
    filepath = "test_user_dual_vector.json"
    user.save_to_json(filepath)
    
    # Load user from JSON
    loaded_user = User.load_from_json(filepath)
    
    # Verify data integrity
    assert loaded_user.name == user.name
    assert loaded_user.interests_profile == user.interests_profile
    assert loaded_user.personality_profile == user.personality_profile
    
    # Verify embeddings
    if user.interests_embedding is not None:
        assert np.array_equal(loaded_user.interests_embedding, user.interests_embedding)
    if user.personality_embedding is not None:
        assert np.array_equal(loaded_user.personality_embedding, user.personality_embedding)
    
    # Clean up
    Path(filepath).unlink()
    
    logger.info("✅ Serialization and loading tests passed!")


def main():
    """Run comprehensive dual-vector workflow tests."""
    logger.info("🚀 Starting comprehensive dual-vector workflow tests...")
    
    start_time = time.time()
    
    try:
        # Initialize generators
        logger.info("Initializing generators...")
        profile_generator = ProfileGenerator()
        embedding_generator = EmbeddingGenerator()
        
        # Test 1: User creation and validation
        user = test_user_creation_and_validation()
        
        # Test 2: Profile generation
        user = test_profile_generation(user, profile_generator)
        
        # Test 3: Embedding generation
        user = test_embedding_generation(user, embedding_generator)
        
        # Test 4: Similarity calculations
        users = test_similarity_calculations()
        
        # Test 5: Population operations
        population = test_population_operations(users)
        
        # Test 6: Compatibility reporting
        test_compatibility_reporting(users)
        
        # Test 7: Serialization and loading
        test_serialization_and_loading(users[0])
        
        # Final validation
        final_validation = validate_dual_vector_user(users[0])
        logger.info(f"Final validation: {final_validation}")
        
        # Performance summary
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("🎉 All tests passed successfully!")
        logger.info(f"⏱️  Total test duration: {duration:.2f} seconds")
        logger.info(f"👥 Processed {len(users)} users")
        logger.info(f"📊 Generated {len(users) * 2} profile types")
        logger.info(f"🔢 Generated {len(users) * 2} embedding vectors")
        logger.info(f"📈 Population size: {len(population)} users")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Dual-vector workflow implementation is working correctly!")
    else:
        print("\n❌ Tests failed!")
        exit(1)