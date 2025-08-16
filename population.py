#!/usr/bin/env python3
"""
Fresh Population class for managing collections of users.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from user import User


class Population:
    """
    Manages a collection of users with similarity search capabilities.
    """
    
    def __init__(self, name: str = "Population"):
        """
        Initialize a population.
        
        Args:
            name: Name of the population
        """
        self.name = name
        self.users = []
        self.logger = logging.getLogger(__name__)
    
    def add_user(self, user: User):
        """Add a user to the population."""
        self.users.append(user)
        self.logger.info(f"Added user {user.name} to population. Total: {len(self.users)}")
    
    def remove_user(self, user: User):
        """Remove a user from the population."""
        if user in self.users:
            self.users.remove(user)
            self.logger.info(f"Removed user {user.name} from population. Total: {len(self.users)}")
    
    def find_similar_users(self, target_user: User, mode: str = 'combined', top_k: int = 5) -> List[Tuple[User, float]]:
        """
        Find most similar users to target user.
        
        Args:
            target_user: User to find similarities for
            mode: 'combined', 'interests', or specific trait name
            top_k: Number of similar users to return
            
        Returns:
            List of (user, similarity_score) tuples, sorted by similarity
        """
        similarities = []
        
        for user in self.users:
            if user == target_user:
                continue
            
            try:
                similarity = target_user.calculate_similarity(user, mode)
                similarities.append((user, similarity))
            except Exception as e:
                self.logger.warning(f"Failed to calculate similarity between {target_user.name} and {user.name}: {e}")
                continue
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_users_with_profiles(self) -> List[User]:
        """Get users that have generated profiles."""
        return [user for user in self.users if user.interests_profile or user.personality_profiles]
    
    def get_users_with_embeddings(self) -> List[User]:
        """Get users that have generated embeddings."""
        return [user for user in self.users if user.get_combined_embedding() is not None]
    
    def get_embedding_matrix(self, mode: str = 'combined') -> Tuple[np.ndarray, List[User]]:
        """
        Get embedding matrix for all users.
        
        Args:
            mode: 'combined', 'interests', or specific trait name
            
        Returns:
            (embedding_matrix, user_list) tuple
        """
        embeddings = []
        valid_users = []
        
        for user in self.users:
            if mode == 'combined':
                emb = user.get_combined_embedding()
            elif mode == 'interests':
                emb = user.interests_embedding
            elif mode == 'Openness':
                emb = user.openness_embedding
            elif mode == 'Conscientiousness':
                emb = user.conscientiousness_embedding
            elif mode == 'Extraversion':
                emb = user.extraversion_embedding
            elif mode == 'Agreeableness':
                emb = user.agreeableness_embedding
            elif mode == 'Neuroticism':
                emb = user.neuroticism_embedding
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            if emb is not None:
                embeddings.append(emb)
                valid_users.append(user)
        
        if not embeddings:
            return np.array([]), []
        
        return np.array(embeddings), valid_users
    
    def save_to_json(self, filepath: str):
        """
        Save population to JSON file.
        
        Args:
            filepath: Path to save file
        """
        data = {
            'name': self.name,
            'users': [user.to_dict() for user in self.users],
            'metadata': {
                'total_users': len(self.users),
                'users_with_profiles': len(self.get_users_with_profiles()),
                'users_with_embeddings': len(self.get_users_with_embeddings())
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved population '{self.name}' with {len(self.users)} users to {filepath}")
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'Population':
        """
        Load population from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Population instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        population = cls(data.get('name', 'Loaded Population'))
        
        for user_data in data.get('users', []):
            user = User.from_dict(user_data)
            population.add_user(user)
        
        return population
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get population statistics."""
        users_with_profiles = self.get_users_with_profiles()
        users_with_embeddings = self.get_users_with_embeddings()
        
        # Check embedding dimensions
        embedding_stats = {}
        if users_with_embeddings:
            sample_user = users_with_embeddings[0]
            
            if sample_user.interests_embedding is not None:
                embedding_stats['interests_dims'] = len(sample_user.interests_embedding)
            
            trait_dims = {}
            for trait in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']:
                emb = getattr(sample_user, f"{trait.lower()}_embedding", None)
                if emb is not None:
                    trait_dims[trait] = len(emb)
            
            embedding_stats['trait_dims'] = trait_dims
            
            combined = sample_user.get_combined_embedding()
            if combined is not None:
                embedding_stats['combined_dims'] = len(combined)
        
        return {
            'name': self.name,
            'total_users': len(self.users),
            'users_with_profiles': len(users_with_profiles),
            'users_with_embeddings': len(users_with_embeddings),
            'embedding_stats': embedding_stats
        }
    
    def create_similarity_matrix(self, mode: str = 'combined') -> Tuple[np.ndarray, List[User]]:
        """
        Create full similarity matrix for the population.
        
        Args:
            mode: Embedding mode to use
            
        Returns:
            (similarity_matrix, user_list) tuple
        """
        embedding_matrix, users = self.get_embedding_matrix(mode)
        
        if len(embedding_matrix) == 0:
            return np.array([]), []
        
        # Calculate cosine similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embedding_matrix)
        
        return similarity_matrix, users
    
    def __len__(self) -> int:
        """Return number of users."""
        return len(self.users)
    
    def __iter__(self):
        """Iterate over users."""
        return iter(self.users)
    
    def __getitem__(self, index) -> User:
        """Get user by index."""
        return self.users[index]
    
    def __str__(self) -> str:
        """String representation."""
        return f"Population('{self.name}', {len(self.users)} users)"