#!/usr/bin/env python3
"""
Fresh User class for unified personality profiling system.
"""

import json
import numpy as np
from typing import Dict, Any, Optional


class User:
    """
    Represents a user with unified personality profiling approach.
    Combines cultural preferences with personality traits for rich embeddings.
    """
    
    def __init__(self, name: str):
        """
        Initialize a user.
        
        Args:
            name: User's name
        """
        self.name = name
        
        # Profile data from frontend JSON
        self.profile_data = {}  # fullName, major, bio, interests, classYear
        self.books_data = {}    # favoriteBooks, bookReviews, bookReflection
        self.movies_data = {}   # favoriteMovies, movieReviews, movieReflection
        self.music_data = {}    # musicArtists, vibeMatch
        self.personality_data = {}  # Big 5 responses with selected/not_selected
        self.additional_info = {}   # talkAboutForHours, perfectDay, etc.
        
        # Generated profiles (unified approach)
        self.interests_profile = None  # 200-word unified personality from cultural data
        self.personality_profiles = {}  # 5 trait descriptions in JSON format
        
        # Embeddings (768D each)
        self.interests_embedding = None     # 768D from interests_profile
        self.openness_embedding = None      # 768D from Openness trait
        self.conscientiousness_embedding = None  # 768D from Conscientiousness trait
        self.extraversion_embedding = None  # 768D from Extraversion trait
        self.agreeableness_embedding = None # 768D from Agreeableness trait
        self.neuroticism_embedding = None   # 768D from Neuroticism trait
        
        # Metadata
        self.metadata = {}
        self.special = False
    
    @classmethod
    def from_json_file(cls, json_path: str) -> 'User':
        """
        Create a user from a JSON file (like yahya_profile.json).
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            User instance
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return cls.from_json_data(data)
    
    @classmethod
    def from_json_data(cls, data: Dict[str, Any]) -> 'User':
        """
        Create a user from JSON data.
        
        Args:
            data: JSON data dictionary
            
        Returns:
            User instance
        """
        # Extract name
        profile = data.get('profile', {})
        name = profile.get('fullName', 'Unknown User')
        
        user = cls(name)
        
        # Store all sections
        user.profile_data = profile
        user.books_data = data.get('books', {})
        user.movies_data = data.get('movies', {})
        user.music_data = data.get('music', {})
        user.personality_data = data.get('personality', {})
        user.additional_info = data.get('additionalInfo', {})
        
        # Store metadata
        user.metadata = {
            'source': 'json_file',
            'original_data': data
        }
        
        return user
    
    def set_interests_profile(self, profile: str):
        """Set the unified interests/personality profile."""
        self.interests_profile = profile
    
    def set_personality_profiles(self, profiles: Dict[str, str]):
        """Set the individual trait profiles."""
        self.personality_profiles = profiles
    
    def set_interests_embedding(self, embedding: np.ndarray):
        """Set the interests embedding (768D)."""
        self.interests_embedding = embedding
    
    def set_trait_embeddings(self, embeddings: Dict[str, np.ndarray]):
        """Set the trait embeddings (5 x 768D)."""
        self.openness_embedding = embeddings.get('Openness')
        self.conscientiousness_embedding = embeddings.get('Conscientiousness')
        self.extraversion_embedding = embeddings.get('Extraversion')
        self.agreeableness_embedding = embeddings.get('Agreeableness')
        self.neuroticism_embedding = embeddings.get('Neuroticism')
    
    def get_combined_embedding(self) -> Optional[np.ndarray]:
        """
        Get combined embedding (interests + 5 traits = 6 x 768D = 4608D).
        
        Returns:
            Combined embedding or None if not all embeddings are available
        """
        embeddings = []
        
        # Add interests embedding
        if self.interests_embedding is not None:
            embeddings.append(self.interests_embedding)
        else:
            return None
        
        # Add trait embeddings in order
        trait_embeddings = [
            self.openness_embedding,
            self.conscientiousness_embedding,
            self.extraversion_embedding,
            self.agreeableness_embedding,
            self.neuroticism_embedding
        ]
        
        for emb in trait_embeddings:
            if emb is not None:
                embeddings.append(emb)
            else:
                return None
        
        return np.concatenate(embeddings)
    
    def calculate_similarity(self, other: 'User', mode: str = 'combined') -> float:
        """
        Calculate similarity with another user using Euclidean distance.
        
        Args:
            other: Another user
            mode: 'combined', 'interests', or specific trait name
            
        Returns:
            Similarity score (0-1, where 1=most similar, 0=least similar)
        """
        if mode == 'combined':
            emb1 = self.get_combined_embedding()
            emb2 = other.get_combined_embedding()
        elif mode == 'interests':
            emb1 = self.interests_embedding
            emb2 = other.interests_embedding
        elif mode == 'Openness':
            emb1 = self.openness_embedding
            emb2 = other.openness_embedding
        elif mode == 'Conscientiousness':
            emb1 = self.conscientiousness_embedding
            emb2 = other.conscientiousness_embedding
        elif mode == 'Extraversion':
            emb1 = self.extraversion_embedding
            emb2 = other.extraversion_embedding
        elif mode == 'Agreeableness':
            emb1 = self.agreeableness_embedding
            emb2 = other.agreeableness_embedding
        elif mode == 'Neuroticism':
            emb1 = self.neuroticism_embedding
            emb2 = other.neuroticism_embedding
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Euclidean distance (L2 norm)
        euclidean_dist = np.sqrt(np.sum((emb1 - emb2) ** 2))
        
        # Convert to similarity score (0-1, where 1=most similar)
        # Normalize by the maximum possible distance between unit vectors
        max_possible_distance = np.sqrt(2 * len(emb1))  # Max distance between normalized vectors
        
        # Normalize embeddings to unit length for consistent comparison
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0  # One of the embeddings is zero
        
        # Normalize embeddings and calculate Euclidean distance
        emb1_normalized = emb1 / norm1
        emb2_normalized = emb2 / norm2
        normalized_euclidean_dist = np.sqrt(np.sum((emb1_normalized - emb2_normalized) ** 2))
        
        # Convert distance to similarity (inverse relationship)
        similarity = 1.0 - (normalized_euclidean_dist / max_possible_distance)
        
        # Ensure the result is between 0 and 1
        return float(max(0.0, min(1.0, similarity)))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary for saving."""
        return {
            'name': self.name,
            'profile_data': self.profile_data,
            'books_data': self.books_data,
            'movies_data': self.movies_data,
            'music_data': self.music_data,
            'personality_data': self.personality_data,
            'additional_info': self.additional_info,
            'interests_profile': self.interests_profile,
            'personality_profiles': self.personality_profiles,
            'interests_embedding': self.interests_embedding.tolist() if self.interests_embedding is not None else None,
            'openness_embedding': self.openness_embedding.tolist() if self.openness_embedding is not None else None,
            'conscientiousness_embedding': self.conscientiousness_embedding.tolist() if self.conscientiousness_embedding is not None else None,
            'extraversion_embedding': self.extraversion_embedding.tolist() if self.extraversion_embedding is not None else None,
            'agreeableness_embedding': self.agreeableness_embedding.tolist() if self.agreeableness_embedding is not None else None,
            'neuroticism_embedding': self.neuroticism_embedding.tolist() if self.neuroticism_embedding is not None else None,
            'metadata': self.metadata,
            'special': self.special
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create user from dictionary."""
        # Extract clean name from potentially multi-line name field
        raw_name = data['name']
        if '\n' in raw_name:
            # If name contains newlines, extract just the first line (the actual name)
            clean_name = raw_name.split('\n')[0].strip()
        else:
            clean_name = raw_name.strip()
        
        user = cls(clean_name)
        
        user.profile_data = data.get('profile_data', {})
        user.books_data = data.get('books_data', {})
        user.movies_data = data.get('movies_data', {})
        user.music_data = data.get('music_data', {})
        user.personality_data = data.get('personality_data', {})
        user.additional_info = data.get('additional_info', {})
        
        user.interests_profile = data.get('interests_profile')
        user.personality_profiles = data.get('personality_profiles', {})
        
        # Load embeddings
        if data.get('interests_embedding'):
            user.interests_embedding = np.array(data['interests_embedding'])
        if data.get('openness_embedding'):
            user.openness_embedding = np.array(data['openness_embedding'])
        if data.get('conscientiousness_embedding'):
            user.conscientiousness_embedding = np.array(data['conscientiousness_embedding'])
        if data.get('extraversion_embedding'):
            user.extraversion_embedding = np.array(data['extraversion_embedding'])
        if data.get('agreeableness_embedding'):
            user.agreeableness_embedding = np.array(data['agreeableness_embedding'])
        if data.get('neuroticism_embedding'):
            user.neuroticism_embedding = np.array(data['neuroticism_embedding'])
        
        user.metadata = data.get('metadata', {})
        user.special = data.get('special', False)
        
        return user
    
    def __str__(self) -> str:
        """String representation."""
        interests_status = "Yes" if self.interests_profile else "No"
        personality_status = f"{len(self.personality_profiles)}/5" if self.personality_profiles else "No"
        
        # Check embeddings
        emb_count = sum([
            self.interests_embedding is not None,
            self.openness_embedding is not None,
            self.conscientiousness_embedding is not None,
            self.extraversion_embedding is not None,
            self.agreeableness_embedding is not None,
            self.neuroticism_embedding is not None
        ])
        
        embedding_status = f"{emb_count}/6"
        special_marker = "⭐" if self.special else ""
        
        return f"User(name='{self.name}', profiles={interests_status}+{personality_status}, embeddings={embedding_status}{special_marker})"