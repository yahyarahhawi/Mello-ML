#!/usr/bin/env python3
"""
Fresh EmbeddingGenerator for 768D interests and 5x768D personality embeddings.
"""

import os
import time
import random
import requests
import numpy as np
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv

from user import User


class EmbeddingGenerator:
    """
    Generates 768D embeddings for interests and each personality trait.
    Uses text-embedding-004 by default for high quality embeddings.
    """
    
    def __init__(self, model_name: str = None, api_key: str = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Embedding model to use
            api_key: Gemini API key
        """
        load_dotenv()
        
        self.model_name = model_name or os.getenv('EMBEDDING_MODEL', 'text-embedding-004')
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
        
        self.logger = logging.getLogger(__name__)
        self.request_count = 0
        self.last_request_time = 0
        
        # Rate limiting
        self.min_delay = 1.0
        self.max_delay = 3.0
        self.max_retries = 3
        self.backoff_factor = 2
        
        # Target dimensions from .env (768D for all embeddings)
        self.target_dims = int(os.getenv('EMBEDDING_DIMENSIONS', '768'))
    
    def _wait_for_rate_limit(self):
        """Implement rate limiting with random delays."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            time.sleep(sleep_time)
        
        # Add random delay
        random_delay = random.uniform(0, self.max_delay - self.min_delay)
        time.sleep(random_delay)
        
        self.last_request_time = time.time()
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate 768D embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            768D numpy array
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:embedContent?key={self.api_key}"
        
        headers = {"Content-Type": "application/json"}
        data = {
            "model": f"models/{self.model_name}",
            "content": {"parts": [{"text": text}]},
            "taskType": "SEMANTIC_SIMILARITY",
            "outputDimensionality": self.target_dims
        }
        
        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()
                
                response = requests.post(url, headers=headers, json=data, timeout=30)
                self.request_count += 1
                
                if response.status_code == 200:
                    result = response.json()
                    embedding = result['embedding']['values']
                    embedding_array = np.array(embedding)
                    
                    # Ensure exactly 768 dimensions
                    if len(embedding_array) != self.target_dims:
                        if len(embedding_array) < self.target_dims:
                            # Pad with zeros
                            padded = np.zeros(self.target_dims)
                            padded[:len(embedding_array)] = embedding_array
                            embedding_array = padded
                        else:
                            # Truncate
                            embedding_array = embedding_array[:self.target_dims]
                    
                    return embedding_array
                
                elif response.status_code == 429:  # Rate limit
                    wait_time = (2 ** attempt) * self.backoff_factor
                    self.logger.warning(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code == 503:  # Server error
                    wait_time = (2 ** attempt) * self.backoff_factor
                    self.logger.warning(f"Server error. Waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}")
                    time.sleep(wait_time)
                    continue
                
                else:
                    self.logger.error(f"API error {response.status_code}: {response.text}")
                    if attempt == self.max_retries - 1:
                        raise Exception(f"API request failed: {response.status_code}")
            
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request exception: {e}")
                if attempt == self.max_retries - 1:
                    raise
                
                wait_time = (2 ** attempt) * self.backoff_factor
                time.sleep(wait_time)
        
        raise Exception(f"Failed to generate embedding after {self.max_retries} attempts")
    
    def embed_user_interests(self, user: User) -> bool:
        """
        Generate 768D interests embedding from unified personality profile.
        
        Args:
            user: User to generate embedding for
            
        Returns:
            True if successful
        """
        if not user.interests_profile:
            self.logger.error(f"User {user.name} has no interests profile")
            return False
        
        try:
            embedding = self.generate_embedding(user.interests_profile)
            user.set_interests_embedding(embedding)
            self.logger.info(f"Generated interests embedding for {user.name} (768D)")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to generate interests embedding for {user.name}: {e}")
            return False
    
    def embed_user_personality_traits(self, user: User) -> bool:
        """
        Generate 768D embeddings for each personality trait.
        
        Args:
            user: User to generate embeddings for
            
        Returns:
            True if successful
        """
        if not user.personality_profiles:
            self.logger.error(f"User {user.name} has no personality profiles")
            return False
        
        trait_embeddings = {}
        traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        
        for trait in traits:
            if trait not in user.personality_profiles:
                self.logger.error(f"Missing {trait} profile for {user.name}")
                return False
            
            trait_text = user.personality_profiles[trait]
            
            try:
                embedding = self.generate_embedding(trait_text)
                trait_embeddings[trait] = embedding
                self.logger.info(f"Generated {trait} embedding for {user.name} (768D)")
            
            except Exception as e:
                self.logger.error(f"Failed to generate {trait} embedding for {user.name}: {e}")
                return False
        
        # Set all trait embeddings
        user.set_trait_embeddings(trait_embeddings)
        self.logger.info(f"Generated all personality trait embeddings for {user.name}")
        return True
    
    def embed_user_complete(self, user: User) -> bool:
        """
        Generate all embeddings for a user (interests + 5 traits).
        
        Args:
            user: User to generate embeddings for
            
        Returns:
            True if all embeddings successful
        """
        interests_success = self.embed_user_interests(user)
        traits_success = self.embed_user_personality_traits(user)
        
        success = interests_success and traits_success
        
        if success:
            self.logger.info(f"Generated complete embeddings for {user.name} (6 x 768D)")
        else:
            self.logger.warning(f"Incomplete embeddings for {user.name}")
        
        return success
    
    def embed_users_batch(self, users: list, progress_callback=None) -> int:
        """
        Generate embeddings for multiple users.
        
        Args:
            users: List of users
            progress_callback: Function to call with progress updates
            
        Returns:
            Number of successfully embedded users
        """
        successful_embeddings = 0
        
        self.logger.info(f"Generating embeddings for {len(users)} users")
        
        for i, user in enumerate(users):
            if self.embed_user_complete(user):
                successful_embeddings += 1
            
            if progress_callback:
                progress_callback(i + 1, len(users))
            
            # Progress update every 10 users
            if (i + 1) % 10 == 0:
                self.logger.info(f"Progress: {successful_embeddings}/{i + 1} users embedded")
        
        self.logger.info(f"Successfully embedded {successful_embeddings}/{len(users)} users")
        return successful_embeddings
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get embedding statistics."""
        return {
            "model_name": self.model_name,
            "target_dimensions": self.target_dims,
            "total_requests": self.request_count,
            "rate_limits": {
                "min_delay": self.min_delay,
                "max_delay": self.max_delay,
                "max_retries": self.max_retries
            }
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"EmbeddingGenerator(model={self.model_name}, dims={self.target_dims}, requests={self.request_count})"