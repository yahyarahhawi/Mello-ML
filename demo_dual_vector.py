#!/usr/bin/env python3
"""
Demonstration script for the dual-vector system using frontend JSON format.
Shows the complete workflow from JSON loading to similarity matching.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any

from user import User
from population import Population
from profile_generator import ProfileGenerator
from embedding_generator import EmbeddingGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_user_from_frontend_json(json_filepath: str) -> User:
    """
    Load a user from frontend JSON format.
    
    Args:
        json_filepath: Path to the JSON file from the React frontend
        
    Returns:
        User object populated with frontend data
    """
    with open(json_filepath, 'r') as f:
        frontend_data = json.load(f)
    
    user = User.from_frontend_json(frontend_data)
    logger.info(f"Loaded user: {user}")
    return user


def generate_dual_profiles_workflow(user: User, profile_generator: ProfileGenerator) -> bool:
    """
    Generate both interests and personality profiles for a user.
    
    Args:
        user: User object loaded from frontend JSON
        profile_generator: ProfileGenerator instance
        
    Returns:
        True if both profiles were generated successfully
    """
    logger.info(f"Generating dual profiles for {user.name}...")
    
    # Generate interests profile
    interests_success = profile_generator.generate_interests_profile(user)
    if interests_success:
        logger.info(f"✅ Interests profile generated (length: {len(user.interests_profile)} chars)")
        print(f"Interests Profile Preview: {user.interests_profile[:150]}...")
    else:
        logger.error(f"❌ Failed to generate interests profile")
    
    # Generate personality profiles
    personality_success = profile_generator.generate_personality_profiles(user)
    if personality_success:
        logger.info(f"✅ Personality profiles generated for {len(user.personality_profile)} traits")
        for trait, description in user.personality_profile.items():
            print(f"{trait}: {description[:100]}...")
    else:
        logger.error(f"❌ Failed to generate personality profiles")
    
    return interests_success and personality_success


def generate_dual_embeddings_workflow(user: User, embedding_generator: EmbeddingGenerator) -> bool:
    """
    Generate both interests and personality embeddings for a user.
    
    Args:
        user: User object with profiles
        embedding_generator: EmbeddingGenerator instance
        
    Returns:
        True if both embeddings were generated successfully
    """
    logger.info(f"Generating dual embeddings for {user.name}...")
    
    success = embedding_generator.embed_user_dual(user)
    
    if success:
        logger.info(f"✅ Dual embeddings generated successfully")
        logger.info(f"   Interests embedding shape: {user.interests_embedding.shape}")
        logger.info(f"   Personality embedding shape: {user.personality_embedding.shape}")
        
        # Calculate combined embedding size
        combined_size = len(user.interests_embedding) + len(user.personality_embedding)
        logger.info(f"   Combined embedding size: {combined_size} dimensions")
        
        return True
    else:
        logger.error(f"❌ Failed to generate dual embeddings")
        return False


def demonstrate_similarity_matching(users: list, mode: str = 'combined'):
    """
    Demonstrate similarity matching with dual-vector system.
    
    Args:
        users: List of User objects with dual embeddings
        mode: Similarity mode ('combined', 'interests', 'personality')
    """
    if len(users) < 2:
        logger.warning("Need at least 2 users for similarity matching")
        return
    
    logger.info(f"Demonstrating {mode} similarity matching...")
    
    target_user = users[0]
    other_users = users[1:]
    
    # Calculate similarities manually
    similarities = []
    for other_user in other_users:
        try:
            similarity = target_user.calculate_similarity(other_user, mode)
            similarities.append((other_user.name, similarity))
        except ValueError as e:
            logger.warning(f"Skipping {other_user.name}: {e}")
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{mode.title()} Similarity Rankings for {target_user.name}:")
    for i, (name, similarity) in enumerate(similarities[:5], 1):
        print(f"  {i}. {name}: {similarity:.3f}")


def create_sample_frontend_json(filepath: str):
    """
    Create a sample frontend JSON file for testing.
    
    Args:
        filepath: Where to save the sample JSON
    """
    sample_data = {
        "profile": {
            "fullName": "Alex Johnson",
            "classYear": "2025",
            "major": "Computer Science",
            "bio": "Love reading sci-fi and building apps. Always down for coffee and deep conversations.",
            "interests": ["Programming", "Science Fiction", "Coffee", "Hiking", "Philosophy"]
        },
        "books": {
            "favoriteBooks": ["Dune", "The Left Hand of Darkness", "Neuromancer", "Foundation", "The Martian"],
            "bookReviews": {
                "Dune": "Epic world-building and political intrigue. Herbert's universe feels incredibly detailed and lived-in.",
                "The Left Hand of Darkness": "Brilliant exploration of gender and society. Le Guin's writing is both beautiful and thought-provoking.",
                "Neuromancer": "Groundbreaking cyberpunk that predicted so much of our digital future."
            },
            "bookReflection": "I'm drawn to science fiction that explores big ideas about humanity, technology, and society. I love books that make me think differently about the world."
        },
        "movies": {
            "favoriteMovies": ["Blade Runner 2049", "Her", "The Social Network", "Arrival", "Ex Machina"],
            "movieReviews": {
                "Blade Runner 2049": "Visually stunning sequel that honors the original while telling its own story.",
                "Her": "Beautiful meditation on love, loneliness, and what makes us human.",
                "Arrival": "Thoughtful sci-fi that puts linguistics and communication at the center."
            },
            "movieReflection": "I prefer films that blend great storytelling with deeper themes about technology and human connection."
        },
        "music": {
            "musicArtists": ["Radiohead", "Bon Iver", "Thom Yorke", "Boards of Canada", "Brian Eno"],
            "vibeMatch": "I love atmospheric, experimental music that creates a mood. Electronic textures mixed with organic sounds."
        },
        "personality": {
            "Openness": {
                "selected": [
                    "I seek out restaurants serving cuisine I've never tried",
                    "I regularly explore music outside my usual genres",
                    "I enjoy unusual or experimental art even if I don't fully understand it",
                    "I prefer exploring new neighborhoods over returning to familiar ones",
                    "I enjoy learning about unfamiliar topics just for fun"
                ],
                "not_selected": [
                    "I stick mostly to familiar authors or genres when reading",
                    "I prefer detailed itineraries when traveling somewhere new",
                    "I use the same apps and rarely try new ones",
                    "I shop at the same stores and buy similar clothes repeatedly",
                    "I rewatch the same TV shows or movies for comfort"
                ],
                "notes": "I love trying new things, but I do have some comfort zones I return to."
            },
            "Conscientiousness": {
                "selected": [
                    "I write down tasks as soon as I think of them",
                    "I make my bed without thinking about it",
                    "I rarely miss deadlines for assignments or work",
                    "I tidy up my space soon after it gets messy",
                    "I use to-do lists and use them to guide my day"
                ],
                "not_selected": [
                    "I have dozens of unread notifications on my phone",
                    "I sometimes run a few minutes late to events",
                    "I leave emails unread for long periods",
                    "I sometimes lose momentum on personal goals after the initial excitement"
                ],
                "notes": "I'm pretty organized but not obsessive about it."
            },
            "Extraversion": {
                "selected": [
                    "I think out loud when working through ideas",
                    "I regularly invite others to hang out",
                    "I sometimes start conversations with strangers",
                    "I feel energized after spending time with groups"
                ],
                "not_selected": [
                    "At parties, I gravitate toward smaller conversations",
                    "I often eat or take breaks alone even when others are around",
                    "Large social events leave me tired afterward",
                    "I enjoy solo hobbies like reading or gaming",
                    "I prefer quiet study spaces over busy ones",
                    "I need alone time to recharge after socializing"
                ],
                "notes": "I'm social but also value my alone time for reflection."
            },
            "Agreeableness": {
                "selected": [
                    "I split bills evenly, even if I ordered less or more",
                    "I'm fine letting others choose the music or activity",
                    "I offer help without waiting to be asked",
                    "I check what everyone wants before ordering for a group",
                    "I apologize when I realize I hurt someone",
                    "I adjust my plans to fit others' schedules"
                ],
                "not_selected": [
                    "I sometimes push for my choice in group decisions",
                    "I wait to trust people until I know them better",
                    "I tell people directly when something bothers me",
                    "I sometimes put my own needs ahead of others' requests"
                ],
                "notes": "I try to be considerate but I'm not a pushover."
            },
            "Neuroticism": {
                "selected": [
                    "I reread emails before sending them to avoid mistakes",
                    "I sometimes lose sleep over upcoming events or deadlines",
                    "I sometimes replay conversations or moments in my head",
                    "I make backup plans in case the first one fails",
                    "I occasionally worry about embarrassing myself"
                ],
                "not_selected": [
                    "I rarely dwell on things I can't control",
                    "I'm usually calm before social events",
                    "I recover quickly after disappointment",
                    "I can laugh off small mistakes easily",
                    "I stay focused in emergencies or high-pressure situations"
                ],
                "notes": "I do worry sometimes but I'm working on being more resilient."
            }
        },
        "additionalInfo": {
            "talkAboutForHours": "The intersection of technology and human psychology. How AI might change society. Philosophy of consciousness.",
            "naturalConversations": "Deep dives into ideas, sharing personal stories, debating ethics of technology, discussing books and movies.",
            "aloneTimeReachFor": "Reading sci-fi, coding personal projects, listening to ambient music, taking long walks.",
            "perfectDay": "Morning coffee while reading, afternoon coding or hiking, evening with friends discussing ideas over dinner.",
            "energySources": ["Meaningful conversations", "Learning new things", "Creative projects", "Nature", "Good music"]
        },
        "metadata": {
            "timestamp": "2024-01-15T10:30:00Z",
            "version": "1.0"
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info(f"Created sample frontend JSON: {filepath}")


def main():
    """Main demonstration workflow."""
    logger.info("🚀 Starting dual-vector system demonstration...")
    
    # Create sample data if it doesn't exist
    sample_json_path = "sample_frontend_data.json"
    if not Path(sample_json_path).exists():
        create_sample_frontend_json(sample_json_path)
    
    try:
        # Initialize generators
        logger.info("Initializing generators...")
        profile_generator = ProfileGenerator()
        embedding_generator = EmbeddingGenerator()
        
        # Load user from frontend JSON
        logger.info("Loading user from frontend JSON...")
        user = load_user_from_frontend_json(sample_json_path)
        
        # Generate dual profiles
        logger.info("Generating dual profiles...")
        profiles_success = generate_dual_profiles_workflow(user, profile_generator)
        
        if not profiles_success:
            logger.error("Failed to generate profiles. Stopping demonstration.")
            return
        
        # Generate dual embeddings
        logger.info("Generating dual embeddings...")
        embeddings_success = generate_dual_embeddings_workflow(user, embedding_generator)
        
        if not embeddings_success:
            logger.error("Failed to generate embeddings. Stopping demonstration.")
            return
        
        # Create a population and add the user
        population = Population("Demo Population")
        population.add_user(user)
        
        # If you have more users, you could demonstrate similarity matching
        # For now, just show the statistics
        logger.info("Population statistics:")
        stats = population.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Save the user to demonstrate serialization
        user.save_to_json("demo_user_dual_vector.json")
        logger.info("Saved user with dual vectors to: demo_user_dual_vector.json")
        
        logger.info("✅ Dual-vector demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()