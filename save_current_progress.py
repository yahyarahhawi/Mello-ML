#!/usr/bin/env python3
"""
Save current population progress to avoid regeneration.
Drag and drop this code into your notebook cell.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path

def save_population_progress(population, filename="population_progress.json"):
    """
    Save current population state including any profiles and embeddings generated so far.
    """
    
    print(f"💾 Saving population progress to {filename}...")
    
    # Create data structure
    saved_data = {
        "metadata": {
            "saved_at": datetime.now().isoformat(),
            "population_name": population.name,
            "total_users": len(population.users),
            "users_with_taste_profiles": 0,
            "users_with_legacy_embeddings": 0,
            "users_with_dual_embeddings": 0
        },
        "users": []
    }
    
    # Process each user
    for user in population.users:
        user_data = {
            "name": user.name,
            "has_taste_profile": bool(user.taste_profile),
            "has_legacy_embedding": bool(user.embedding is not None),
            "has_interests_profile": bool(user.interests_profile),
            "has_personality_profile": bool(user.personality_profile),
            "has_interests_embedding": bool(user.interests_embedding is not None),
            "has_personality_embedding": bool(user.personality_embedding is not None),
        }
        
        # Save taste profile if exists
        if user.taste_profile:
            user_data["taste_profile"] = user.taste_profile
            saved_data["metadata"]["users_with_taste_profiles"] += 1
        
        # Save legacy embedding if exists
        if user.embedding is not None:
            user_data["legacy_embedding"] = user.embedding.tolist() if isinstance(user.embedding, np.ndarray) else user.embedding
            saved_data["metadata"]["users_with_legacy_embeddings"] += 1
        
        # Save interests profile and embedding if they exist
        if user.interests_profile:
            user_data["interests_profile"] = user.interests_profile
        
        if user.interests_embedding is not None:
            user_data["interests_embedding"] = user.interests_embedding.tolist()
        
        # Save personality profile and embedding if they exist
        if user.personality_profile:
            user_data["personality_profile"] = user.personality_profile
        
        if user.personality_embedding is not None:
            user_data["personality_embedding"] = user.personality_embedding.tolist()
        
        # Check if user has dual embeddings
        if user.interests_embedding is not None and user.personality_embedding is not None:
            saved_data["metadata"]["users_with_dual_embeddings"] += 1
        
        # Save frontend data if it exists
        if hasattr(user, 'metadata') and 'frontend_data' in user.metadata:
            user_data["frontend_data"] = user.metadata['frontend_data']
        
        # Save any other metadata
        if hasattr(user, 'metadata'):
            user_data["metadata"] = user.metadata
        
        saved_data["users"].append(user_data)
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(saved_data, f, indent=2)
    
    # Print summary
    print(f"✅ Saved {len(population.users)} users to {filename}")
    print(f"📊 Summary:")
    print(f"   Users with taste profiles: {saved_data['metadata']['users_with_taste_profiles']}")
    print(f"   Users with legacy embeddings: {saved_data['metadata']['users_with_legacy_embeddings']}")
    print(f"   Users with dual embeddings: {saved_data['metadata']['users_with_dual_embeddings']}")
    
    return filename


def load_population_progress(filename="population_progress.json"):
    """
    Load previously saved population progress.
    """
    
    print(f"📂 Loading population progress from {filename}...")
    
    if not Path(filename).exists():
        print(f"❌ File {filename} not found!")
        return None
    
    with open(filename, 'r') as f:
        saved_data = json.load(f)
    
    # Create new population
    from population import Population
    from user import User
    
    population = Population(saved_data["metadata"]["population_name"])
    
    # Recreate users
    for user_data in saved_data["users"]:
        user = User(name=user_data["name"])
        
        # Restore taste profile
        if "taste_profile" in user_data:
            user.set_taste_profile(user_data["taste_profile"])
        
        # Restore legacy embedding
        if "legacy_embedding" in user_data:
            user.set_embedding(np.array(user_data["legacy_embedding"]))
        
        # Restore interests profile and embedding
        if "interests_profile" in user_data:
            user.interests_profile = user_data["interests_profile"]
        
        if "interests_embedding" in user_data:
            user.interests_embedding = np.array(user_data["interests_embedding"])
        
        # Restore personality profile and embedding
        if "personality_profile" in user_data:
            user.personality_profile = user_data["personality_profile"]
        
        if "personality_embedding" in user_data:
            user.personality_embedding = np.array(user_data["personality_embedding"])
        
        # Restore frontend data
        if "frontend_data" in user_data:
            if not hasattr(user, 'metadata'):
                user.metadata = {}
            user.metadata['frontend_data'] = user_data["frontend_data"]
        
        # Restore other metadata
        if "metadata" in user_data:
            if not hasattr(user, 'metadata'):
                user.metadata = {}
            user.metadata.update(user_data["metadata"])
        
        population.add_user(user)
    
    print(f"✅ Loaded {len(population.users)} users")
    print(f"📊 Statistics: {population.get_statistics()}")
    
    return population


# ===== DRAG AND DROP CODE FOR YOUR NOTEBOOK =====

# Save current progress
save_population_progress(population, "current_population_progress.json")

# If you want to load it later, use:
# population = load_population_progress("current_population_progress.json")