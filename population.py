#!/usr/bin/env python3
"""
Population class for managing collections of users and performing operations on them.
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from user import User


class Population:
    """
    Manages a collection of User objects and provides methods for analysis,
    similarity search, and bulk operations.
    """
    
    def __init__(self, name: str = "Default Population"):
        """
        Initialize a new population.
        
        Args:
            name: Name identifier for this population
        """
        self.name = name
        self.users: List[User] = []
        self.logger = logging.getLogger(__name__)
    
    def add_user(self, user: User):
        """Add a user to the population."""
        if not isinstance(user, User):
            raise TypeError("Only User objects can be added to population")
        
        # Check for duplicate names
        if any(existing.name == user.name for existing in self.users):
            self.logger.warning(f"User with name '{user.name}' already exists in population")
        
        self.users.append(user)
    
    def remove_user(self, name: str) -> bool:
        """
        Remove a user by name.
        
        Args:
            name: Name of user to remove
            
        Returns:
            True if user was found and removed, False otherwise
        """
        for i, user in enumerate(self.users):
            if user.name == name:
                del self.users[i]
                return True
        return False
    
    def get_user(self, name: str) -> Optional[User]:
        """Get a user by name."""
        for user in self.users:
            if user.name == name:
                return user
        return None
    
    def get_users_with_embeddings(self, mode: str = 'any') -> List[User]:
        """Get all users that have embeddings.
        
        Args:
            mode: 'any', 'legacy', 'interests', 'personality', 'dual', 'combined'
        """
        if mode == 'legacy':
            return [user for user in self.users if hasattr(user, 'embedding') and user.embedding is not None]
        elif mode == 'interests':
            return [user for user in self.users if hasattr(user, 'interests_embedding') and user.interests_embedding is not None]
        elif mode == 'personality':
            return [user for user in self.users if hasattr(user, 'personality_embedding') and user.personality_embedding is not None]
        elif mode == 'dual':
            return [user for user in self.users if hasattr(user, 'interests_embedding') and user.interests_embedding is not None and hasattr(user, 'personality_embedding') and user.personality_embedding is not None]
        elif mode == 'combined':
            return [user for user in self.users if hasattr(user, 'interests_embedding') and user.interests_embedding is not None and hasattr(user, 'personality_embedding') and user.personality_embedding is not None]
        else:  # 'any'
            return [user for user in self.users if ((hasattr(user, 'embedding') and user.embedding is not None) or 
                                                   (hasattr(user, 'interests_embedding') and user.interests_embedding is not None) or 
                                                   (hasattr(user, 'personality_embedding') and user.personality_embedding is not None))]
    
    def get_users_with_taste_profiles(self) -> List[User]:
        """Get all users that have taste profiles."""
        return [user for user in self.users if hasattr(user, 'taste_profile') and user.taste_profile is not None]
    
    def get_users_with_dual_profiles(self) -> List[User]:
        """Get all users that have both interests and personality profiles."""
        return [user for user in self.users if hasattr(user, 'interests_profile') and user.interests_profile is not None and hasattr(user, 'personality_profile') and user.personality_profile is not None]
    
    def get_special_users(self) -> List[User]:
        """Get all users marked as special."""
        return [user for user in self.users if user.special]
    
    def get_regular_users(self) -> List[User]:
        """Get all users not marked as special."""
        return [user for user in self.users if not user.special]
    
    def size(self) -> int:
        """Get the number of users in the population."""
        return len(self.users)
    
    def get_embedding_matrix(self, mode: str = 'legacy') -> np.ndarray:
        """
        Get embeddings as a matrix.
        
        Args:
            mode: 'legacy', 'interests', 'personality', 'combined'
        
        Returns:
            numpy array of shape (n_users, embedding_dim)
        """
        users_with_embeddings = self.get_users_with_embeddings(mode)
        if not users_with_embeddings:
            raise ValueError(f"No users have {mode} embeddings")
        
        embeddings = []
        for user in users_with_embeddings:
            if mode == 'legacy':
                embeddings.append(user.embedding)
            elif mode == 'interests':
                embeddings.append(user.interests_embedding)
            elif mode == 'personality':
                embeddings.append(user.personality_embedding)
            elif mode == 'combined':
                combined = np.concatenate([user.interests_embedding, user.personality_embedding])
                embeddings.append(combined)
        
        return np.array(embeddings)
    
    def get_embedding_names(self, mode: str = 'any') -> List[str]:
        """Get names of users that have embeddings.
        
        Args:
            mode: 'any', 'legacy', 'interests', 'personality', 'dual', 'combined'
        """
        return [user.name for user in self.get_users_with_embeddings(mode)]
    
    def find_similar_users(self, target_user: User, top_k: int = 5, mode: str = 'combined') -> List[Tuple[User, float]]:
        """
        Find the most similar users to a target user.
        
        Args:
            target_user: User to find similarities for
            top_k: Number of similar users to return
            mode: 'combined', 'interests', 'personality', or 'legacy'
            
        Returns:
            List of (User, similarity_score) tuples, sorted by similarity (descending)
        """
        # Check if target user has required embeddings
        if mode == 'legacy' and target_user.embedding is None:
            raise ValueError("Target user must have a legacy embedding")
        elif mode == 'interests' and target_user.interests_embedding is None:
            raise ValueError("Target user must have an interests embedding")
        elif mode == 'personality' and target_user.personality_embedding is None:
            raise ValueError("Target user must have a personality embedding")
        elif mode == 'combined' and (target_user.interests_embedding is None or target_user.personality_embedding is None):
            raise ValueError("Target user must have both interests and personality embeddings")
        
        users_with_embeddings = self.get_users_with_embeddings(mode)
        similarities = []
        
        for user in users_with_embeddings:
            if user.name != target_user.name:  # Don't include the target user
                try:
                    similarity = target_user.calculate_similarity(user, mode)
                    similarities.append((user, similarity))
                except ValueError:
                    # Skip users that don't have the required embedding type
                    continue
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def find_similar_users_by_embedding(self, target_embedding: np.ndarray, top_k: int = 5, mode: str = 'legacy') -> List[Tuple[User, float]]:
        """
        Find users similar to a given embedding vector.
        
        Args:
            target_embedding: Vector to find similarities for
            top_k: Number of similar users to return
            mode: 'legacy', 'interests', 'personality', 'combined'
            
        Returns:
            List of (User, similarity_score) tuples, sorted by similarity (descending)
        """
        users_with_embeddings = self.get_users_with_embeddings(mode)
        similarities = []
        
        # Normalize target embedding
        target_norm = np.linalg.norm(target_embedding)
        if target_norm == 0:
            raise ValueError("Target embedding cannot be zero vector")
        
        for user in users_with_embeddings:
            # Get the appropriate embedding
            if mode == 'legacy':
                user_embedding = user.embedding
            elif mode == 'interests':
                user_embedding = user.interests_embedding
            elif mode == 'personality':
                user_embedding = user.personality_embedding
            elif mode == 'combined':
                user_embedding = np.concatenate([user.interests_embedding, user.personality_embedding])
            
            # Calculate cosine similarity
            if user_embedding is not None:
                user_norm = np.linalg.norm(user_embedding)
                if user_norm > 0:
                    similarity = np.dot(target_embedding, user_embedding) / (target_norm * user_norm)
                    similarities.append((user, float(similarity)))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get population statistics."""
        users_with_legacy_embeddings = self.get_users_with_embeddings('legacy')
        users_with_interests_embeddings = self.get_users_with_embeddings('interests')
        users_with_personality_embeddings = self.get_users_with_embeddings('personality')
        users_with_dual_embeddings = self.get_users_with_embeddings('dual')
        users_with_taste_profiles = self.get_users_with_taste_profiles()
        users_with_dual_profiles = self.get_users_with_dual_profiles()
        special_users = self.get_special_users()
        regular_users = self.get_regular_users()
        
        # Book statistics
        total_books = sum(len(user.preferences.books) for user in self.users)
        avg_books_per_user = total_books / len(self.users) if self.users else 0
        
        # Rating statistics
        all_ratings = []
        for user in self.users:
            all_ratings.extend([book.rating for book in user.preferences.books])
        
        stats = {
            "total_users": len(self.users),
            "special_users": len(special_users),
            "regular_users": len(regular_users),
            "users_with_legacy_embeddings": len(users_with_legacy_embeddings),
            "users_with_interests_embeddings": len(users_with_interests_embeddings),
            "users_with_personality_embeddings": len(users_with_personality_embeddings),
            "users_with_dual_embeddings": len(users_with_dual_embeddings),
            "users_with_taste_profiles": len(users_with_taste_profiles),
            "users_with_dual_profiles": len(users_with_dual_profiles),
            "total_books_rated": total_books,
            "avg_books_per_user": avg_books_per_user,
        }
        
        # Add embedding dimensions if available
        if users_with_legacy_embeddings:
            stats["legacy_embedding_dimension"] = len(users_with_legacy_embeddings[0].embedding)
        if users_with_interests_embeddings:
            stats["interests_embedding_dimension"] = len(users_with_interests_embeddings[0].interests_embedding)
        if users_with_personality_embeddings:
            stats["personality_embedding_dimension"] = len(users_with_personality_embeddings[0].personality_embedding)
        
        if all_ratings:
            stats.update({
                "avg_rating": np.mean(all_ratings),
                "min_rating": min(all_ratings),
                "max_rating": max(all_ratings),
                "rating_std": np.std(all_ratings)
            })
        
        return stats
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert population to pandas DataFrame for analysis."""
        data = []
        for user in self.users:
            user_data = {
                "name": user.name,
                "special": user.special,
                "num_books": len(user.preferences.books),
                "num_movies": len(user.preferences.movies),
                "num_music": len(user.preferences.music),
                "has_taste_profile": user.taste_profile is not None,
                "has_interests_profile": user.interests_profile is not None,
                "has_personality_profile": user.personality_profile is not None,
                "has_legacy_embedding": user.embedding is not None,
                "has_interests_embedding": user.interests_embedding is not None,
                "has_personality_embedding": user.personality_embedding is not None,
                "has_dual_embeddings": user.interests_embedding is not None and user.personality_embedding is not None,
                "avg_book_rating": np.mean([book.rating for book in user.preferences.books]) if user.preferences.books else None
            }
            
            if user.preferences.books:
                user_data["favorite_genres"] = list(set(book.genre for book in user.preferences.books if book.genre))
            
            data.append(user_data)
        
        return pd.DataFrame(data)
    
    def save_to_json(self, filepath: str):
        """Save the entire population to a JSON file."""
        data = {
            "name": self.name,
            "users": [user.to_dict() for user in self.users],
            "metadata": {
                "total_users": len(self.users),
                "created_at": pd.Timestamp.now().isoformat()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved population '{self.name}' with {len(self.users)} users to {filepath}")
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'Population':
        """Load a population from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        population = cls(name=data.get("name", "Loaded Population"))
        
        for user_data in data.get("users", []):
            user = User.from_dict(user_data)
            population.add_user(user)
        
        logging.getLogger(__name__).info(f"Loaded population '{population.name}' with {len(population.users)} users")
        return population
    
    def export_embeddings_only(self, filepath: str, mode: str = 'legacy'):
        """Export only embeddings and names.
        
        Args:
            filepath: Where to save the file
            mode: 'legacy', 'interests', 'personality', 'dual'
        """
        users_with_embeddings = self.get_users_with_embeddings(mode)
        data = []
        
        for user in users_with_embeddings:
            user_data = {"name": user.name}
            
            if mode == 'legacy':
                user_data.update({
                    "books_vector": user.embedding.tolist(),
                    "book_taste": user.taste_profile
                })
            elif mode == 'interests':
                user_data.update({
                    "interests_vector": user.interests_embedding.tolist(),
                    "interests_profile": user.interests_profile
                })
            elif mode == 'personality':
                user_data.update({
                    "personality_vector": user.personality_embedding.tolist(),
                    "personality_profile": user.personality_profile
                })
            elif mode == 'dual':
                user_data.update({
                    "interests_vector": user.interests_embedding.tolist(),
                    "personality_vector": user.personality_embedding.tolist(),
                    "interests_profile": user.interests_profile,
                    "personality_profile": user.personality_profile
                })
            
            data.append(user_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def merge_population(self, other_population: 'Population', handle_duplicates: str = "skip"):
        """
        Merge another population into this one.
        
        Args:
            other_population: Population to merge
            handle_duplicates: "skip", "replace", or "rename"
        """
        for user in other_population.users:
            existing_user = self.get_user(user.name)
            
            if existing_user is None:
                self.add_user(user)
            elif handle_duplicates == "skip":
                continue
            elif handle_duplicates == "replace":
                self.remove_user(user.name)
                self.add_user(user)
            elif handle_duplicates == "rename":
                # Find a unique name
                counter = 1
                new_name = f"{user.name}_{counter}"
                while self.get_user(new_name) is not None:
                    counter += 1
                    new_name = f"{user.name}_{counter}"
                user.name = new_name
                self.add_user(user)
    
    def split_population(self, ratio: float = 0.8, shuffle: bool = True) -> Tuple['Population', 'Population']:
        """
        Split population into two populations (e.g., for train/test).
        
        Args:
            ratio: Fraction for first population
            shuffle: Whether to shuffle before splitting
            
        Returns:
            Tuple of (first_population, second_population)
        """
        users_copy = self.users.copy()
        
        if shuffle:
            np.random.shuffle(users_copy)
        
        split_idx = int(len(users_copy) * ratio)
        
        pop1 = Population(f"{self.name}_split1")
        pop2 = Population(f"{self.name}_split2")
        
        for user in users_copy[:split_idx]:
            pop1.add_user(user)
        
        for user in users_copy[split_idx:]:
            pop2.add_user(user)
        
        return pop1, pop2
    
    def __len__(self) -> int:
        """Return the number of users."""
        return len(self.users)
    
    def __iter__(self):
        """Make population iterable."""
        return iter(self.users)
    
    def __str__(self) -> str:
        """String representation of the population."""
        return f"Population(name='{self.name}', users={len(self.users)})"
    
    def __repr__(self) -> str:
        return self.__str__()