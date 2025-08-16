#!/usr/bin/env python3
"""
Streamlined synthetic data generation script for Mello ML.
Uses structured output and dual-vector pipeline for reliable data creation.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from user import User
from population import Population
from profile_generator import ProfileGenerator
from embedding_generator import EmbeddingGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_synthetic_population(
    size: int = 20,
    save_json: bool = True,
    save_profiles: bool = True,
    save_embeddings: bool = True,
    output_dir: str = "synthetic_data"
) -> Population:
    """
    Generate a complete synthetic population with dual-vector embeddings.
    
    Args:
        size: Number of synthetic users to generate
        save_json: Whether to save raw JSON data
        save_profiles: Whether to save generated profiles
        save_embeddings: Whether to save embeddings
        output_dir: Directory to save outputs
        
    Returns:
        Population object with processed users
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info(f"🚀 Starting synthetic data generation for {size} users")
    logger.info(f"📁 Output directory: {output_path.absolute()}")
    
    # Initialize generators
    logger.info("🤖 Initializing AI generators...")
    profile_generator = ProfileGenerator()
    embedding_generator = EmbeddingGenerator()
    
    logger.info(f"📝 Profile Generator: {profile_generator.taste_profile_model}")
    logger.info(f"🔢 Embedding Generator: {embedding_generator.model_name}")
    
    # Create population
    population = Population(f"Synthetic Population ({size} users)")
    
    # Track generation statistics
    stats = {
        "total_requested": size,
        "json_generated": 0,
        "profiles_generated": 0,
        "embeddings_generated": 0,
        "fully_processed": 0,
        "errors": []
    }
    
    generated_data = {
        "users": [],
        "profiles": {},
        "embeddings": {},
        "metadata": {
            "generation_time": None,
            "generator_versions": {
                "profile_model": profile_generator.taste_profile_model,
                "embedding_model": embedding_generator.model_name,
                "embedding_dimensions": embedding_generator.embedding_dimensions
            }
        }
    }
    
    start_time = time.time()
    
    # Generate users one by one
    logger.info(f"\n🎭 Generating {size} synthetic users...")
    
    for i in range(size):
        user_start_time = time.time()
        logger.info(f"\n👤 User {i+1}/{size}: Generating...")
        
        try:
            # Step 1: Generate synthetic user with JSON structure
            synthetic_user = profile_generator.generate_synthetic_user()
            
            if not synthetic_user:
                stats["errors"].append(f"User {i+1}: Failed to generate base user")
                logger.error(f"❌ Failed to generate synthetic user {i+1}")
                continue
            
            stats["json_generated"] += 1
            logger.info(f"   ✅ Generated JSON structure for {synthetic_user.name}")
            
            # Save raw JSON data
            if save_json and synthetic_user.metadata.get('frontend_data'):
                user_json = synthetic_user.metadata['frontend_data']
                generated_data["users"].append({
                    "id": i+1,
                    "name": synthetic_user.name,
                    "data": user_json
                })
            
            # Step 2: Generate dual profiles (interests + personality)
            logger.info(f"   📖 Generating dual profiles...")
            profile_success = profile_generator.generate_dual_profiles(synthetic_user)
            
            if not profile_success:
                stats["errors"].append(f"User {i+1}: Failed to generate dual profiles")
                logger.error(f"   ❌ Failed to generate profiles for {synthetic_user.name}")
                continue
            
            stats["profiles_generated"] += 1
            logger.info(f"   ✅ Generated interests ({len(synthetic_user.interests_profile)} chars) and personality profiles")
            
            # Save profiles
            if save_profiles:
                generated_data["profiles"][synthetic_user.name] = {
                    "interests_profile": synthetic_user.interests_profile,
                    "personality_profile": synthetic_user.personality_profile
                }
            
            # Step 3: Generate dual embeddings (3072 + 3840 dims)
            logger.info(f"   🔢 Generating dual embeddings...")
            embedding_success = embedding_generator.embed_user_dual(synthetic_user)
            
            if not embedding_success:
                stats["errors"].append(f"User {i+1}: Failed to generate dual embeddings")
                logger.error(f"   ❌ Failed to generate embeddings for {synthetic_user.name}")
                continue
            
            stats["embeddings_generated"] += 1
            logger.info(f"   ✅ Generated embeddings: interests {synthetic_user.interests_embedding.shape}, personality {synthetic_user.personality_embedding.shape}")
            
            # Save embeddings (convert to lists for JSON serialization)
            if save_embeddings:
                generated_data["embeddings"][synthetic_user.name] = {
                    "interests_embedding": synthetic_user.interests_embedding.tolist(),
                    "personality_embedding": synthetic_user.personality_embedding.tolist()
                }
            
            # Step 4: Add to population
            population.add_user(synthetic_user)
            stats["fully_processed"] += 1
            
            user_time = time.time() - user_start_time
            logger.info(f"   🎯 Successfully processed {synthetic_user.name} in {user_time:.1f}s")
            
        except Exception as e:
            error_msg = f"User {i+1}: Unexpected error - {str(e)}"
            stats["errors"].append(error_msg)
            logger.error(f"   ❌ Error processing user {i+1}: {e}")
    
    # Final statistics and saving
    end_time = time.time()
    total_time = end_time - start_time
    generated_data["metadata"]["generation_time"] = total_time
    
    logger.info(f"\n📊 Generation Complete!")
    logger.info(f"   Total time: {total_time:.1f} seconds")
    logger.info(f"   Users with JSON: {stats['json_generated']}/{stats['total_requested']}")
    logger.info(f"   Users with profiles: {stats['profiles_generated']}/{stats['total_requested']}")
    logger.info(f"   Users with embeddings: {stats['embeddings_generated']}/{stats['total_requested']}")
    logger.info(f"   Fully processed: {stats['fully_processed']}/{stats['total_requested']}")
    logger.info(f"   Success rate: {(stats['fully_processed']/stats['total_requested'])*100:.1f}%")
    
    # Save all data
    if save_json or save_profiles or save_embeddings:
        output_file = output_path / "synthetic_population.json"
        with open(output_file, 'w') as f:
            json.dump(generated_data, f, indent=2)
        logger.info(f"💾 Saved complete dataset: {output_file}")
    
    # Save population stats
    pop_stats = population.get_statistics()
    stats_file = output_path / "population_stats.json"
    with open(stats_file, 'w') as f:
        json.dump({
            "population_statistics": pop_stats,
            "generation_statistics": stats,
            "generation_time": total_time
        }, f, indent=2)
    logger.info(f"📈 Saved statistics: {stats_file}")
    
    # Log any errors
    if stats["errors"]:
        logger.warning(f"⚠️  {len(stats['errors'])} errors occurred:")
        for error in stats["errors"][:5]:  # Show first 5 errors
            logger.warning(f"   {error}")
        if len(stats["errors"]) > 5:
            logger.warning(f"   ... and {len(stats['errors']) - 5} more")
    
    return population


def load_synthetic_population(data_file: str = "synthetic_data/synthetic_population.json") -> Optional[Population]:
    """
    Load a previously generated synthetic population from file.
    
    Args:
        data_file: Path to the saved population data
        
    Returns:
        Population object with loaded users, or None if failed
    """
    
    data_path = Path(data_file)
    if not data_path.exists():
        logger.error(f"❌ Data file not found: {data_path}")
        return None
    
    logger.info(f"📂 Loading synthetic population from {data_path}")
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        population = Population("Loaded Synthetic Population")
        
        # Initialize generators for processing
        profile_generator = ProfileGenerator()
        embedding_generator = EmbeddingGenerator()
        
        loaded_count = 0
        
        for user_data in data.get("users", []):
            try:
                # Create user from JSON data
                user = User.from_frontend_json(user_data["data"])
                
                # Load profiles if available
                if user.name in data.get("profiles", {}):
                    profile_data = data["profiles"][user.name]
                    user.interests_profile = profile_data.get("interests_profile")
                    user.personality_profile = profile_data.get("personality_profile")
                
                # Load embeddings if available
                if user.name in data.get("embeddings", {}):
                    embedding_data = data["embeddings"][user.name]
                    import numpy as np
                    user.interests_embedding = np.array(embedding_data["interests_embedding"])
                    user.personality_embedding = np.array(embedding_data["personality_embedding"])
                
                population.add_user(user)
                loaded_count += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to load user {user_data.get('name', 'unknown')}: {e}")
        
        logger.info(f"✅ Loaded {loaded_count} users from {data_path}")
        logger.info(f"📊 Population stats: {population.get_statistics()}")
        
        return population
        
    except Exception as e:
        logger.error(f"❌ Failed to load population data: {e}")
        return None


def main():
    """Generate synthetic data with command line interface."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic user data for Mello ML")
    parser.add_argument("--size", type=int, default=20, help="Number of users to generate")
    parser.add_argument("--output-dir", type=str, default="synthetic_data", help="Output directory")
    parser.add_argument("--load", type=str, help="Load existing data file instead of generating")
    
    args = parser.parse_args()
    
    if args.load:
        # Load existing data
        population = load_synthetic_population(args.load)
        if population:
            print(f"✅ Successfully loaded {len(population)} users")
            print(f"📊 Statistics: {population.get_statistics()}")
        else:
            print("❌ Failed to load data")
            return
    else:
        # Generate new data
        print(f"🚀 Generating {args.size} synthetic users...")
        print(f"📁 Output: {args.output_dir}")
        
        population = generate_synthetic_population(
            size=args.size,
            output_dir=args.output_dir
        )
        
        print(f"\n🎉 Generation complete!")
        print(f"👥 Created population with {len(population)} users")
        print(f"📊 Full statistics saved to {args.output_dir}/population_stats.json")


if __name__ == "__main__":
    main()