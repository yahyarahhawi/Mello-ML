#!/usr/bin/env python3
"""
EmbeddingGenerator class for handling API interactions and embedding generation.
Supports both OpenAI and Google Gemini embedding models.
"""

import os
import time
import random
import requests
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
from dotenv import load_dotenv

from user import User
from population import Population


class EmbeddingGenerator:
    """
    Handles embedding generation using various APIs with robust error handling,
    rate limiting, and retry logic.
    """
    
    def __init__(self, model_name: str = None, api_key: str = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Embedding model to use (defaults to env variable)
            api_key: API key (defaults to env variable)
        """
        load_dotenv()
        
        # Determine which model and API to use
        self.model_name = model_name or os.getenv('EMBEDDING_MODEL', 'gemini-embedding-001')

        
        # Use Gemini API for all models (including text-embedding-004)
        self.provider = 'gemini'
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.embedding_dimensions = int(os.getenv('EMBEDDING_DIMENSIONS', '768'))
        
        if not self.api_key:
            raise ValueError("API key not found. Set GEMINI_API_KEY environment variable.")
        
        self.logger = logging.getLogger(__name__)
        self.request_count = 0
        self.last_request_time = 0
        
        # Rate limiting settings
        self.min_delay = 1.0  # Minimum seconds between requests
        self.max_delay = 3.0  # Maximum seconds between requests
        self.max_retries = 3
        self.backoff_factor = 2
    
    def _wait_for_rate_limit(self):
        """Implement rate limiting with random delays."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            time.sleep(sleep_time)
        
        # Add random delay to avoid synchronized requests
        random_delay = random.uniform(0, self.max_delay - self.min_delay)
        time.sleep(random_delay)
        
        self.last_request_time = time.time()
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text with retry logic.
        
        Args:
            text: Text to embed
            
        Returns:
            numpy array containing the embedding
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        return self._generate_gemini_embedding(text)
    
    def _generate_gemini_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using Google Gemini API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:embedContent?key={self.api_key}"
        
        headers = {"Content-Type": "application/json"}
        data = {
            "model": f"models/{self.model_name}",
            "content": {"parts": [{"text": text}]},
            "taskType": "SEMANTIC_SIMILARITY",
            "outputDimensionality": self.embedding_dimensions
        }
        
        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()
                
                response = requests.post(url, headers=headers, json=data, timeout=30)
                self.request_count += 1
                
                if response.status_code == 200:
                    result = response.json()
                    embedding = result['embedding']['values']
                    return np.array(embedding)
                
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
                    self.logger.error(f"Gemini API error {response.status_code}: {response.text}")
                    if attempt == self.max_retries - 1:
                        raise Exception(f"Gemini API request failed: {response.status_code} - {response.text}")
            
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request exception: {e}")
                if attempt == self.max_retries - 1:
                    raise
                
                wait_time = (2 ** attempt) * self.backoff_factor
                time.sleep(wait_time)
        
        raise Exception(f"Failed to generate Gemini embedding after {self.max_retries} attempts")
    
    def generate_embeddings_batch(self, texts: List[str], progress_callback=None) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            progress_callback: Function to call with progress updates
            
        Returns:
            List of numpy arrays containing embeddings
        """
        embeddings = []
        total = len(texts)
        
        self.logger.info(f"Generating embeddings for {total} texts using {self.model_name}")
        
        for i, text in enumerate(texts):
            try:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
                
                if progress_callback:
                    progress_callback(i + 1, total)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Generated {i + 1}/{total} embeddings")
            
            except Exception as e:
                self.logger.error(f"Failed to generate embedding for text {i}: {e}")
                # Add zero vector as placeholder
                embeddings.append(np.zeros(self.embedding_dimensions))
        
        return embeddings
    
    def embed_user(self, user: User) -> bool:
        """
        Generate and set embedding for a user based on their taste profile.
        
        Args:
            user: User object to generate embedding for
            
        Returns:
            True if successful, False otherwise
        """
        if not user.taste_profile:
            self.logger.error(f"User {user.name} has no taste profile to embed")
            return False
        
        try:
            embedding = self.generate_embedding(user.taste_profile)
            user.set_embedding(embedding)
            self.logger.info(f"Generated embedding for user {user.name}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to generate embedding for user {user.name}: {e}")
            return False
    
    def embed_population(self, population: Population, progress_callback=None) -> int:
        """
        Generate embeddings for all users in a population that have taste profiles.
        
        Args:
            population: Population to generate embeddings for
            progress_callback: Function to call with progress updates
            
        Returns:
            Number of successfully embedded users
        """
        users_with_profiles = population.get_users_with_taste_profiles()
        
        if not users_with_profiles:
            self.logger.warning("No users with taste profiles found")
            return 0
        
        self.logger.info(f"Generating embeddings for {len(users_with_profiles)} users")
        
        successful_embeddings = 0
        
        for i, user in enumerate(users_with_profiles):
            if self.embed_user(user):
                successful_embeddings += 1
            
            if progress_callback:
                progress_callback(i + 1, len(users_with_profiles))
        
        self.logger.info(f"Successfully generated {successful_embeddings}/{len(users_with_profiles)} embeddings")
        return successful_embeddings
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Simple wrapper for generating a single embedding.
        
        Args:
            text: Text to embed
            
        Returns:
            numpy array containing the embedding
        """
        return self.generate_embedding(text)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about API usage."""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "embedding_dimensions": self.embedding_dimensions,
            "total_requests": self.request_count,
            "min_delay": self.min_delay,
            "max_delay": self.max_delay,
            "max_retries": self.max_retries
        }
    
    def reset_statistics(self):
        """Reset request counter."""
        self.request_count = 0
    
    def __str__(self) -> str:
        """String representation."""
        return f"EmbeddingGenerator(provider={self.provider}, model={self.model_name}, dims={self.embedding_dimensions}, requests={self.request_count})"