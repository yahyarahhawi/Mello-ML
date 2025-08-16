#!/usr/bin/env python3
"""
User class for representing individual users with their preferences and embeddings.
"""

import json
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class BookRating:
    """Represents a book with its rating."""
    title: str
    author: str
    rating: float
    genre: Optional[str] = None


@dataclass
class UserPreferences:
    """Container for all user preferences."""
    books: List[BookRating]
    movies: List[Dict[str, Any]] = None
    music: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.movies is None:
            self.movies = []
        if self.music is None:
            self.music = []


class User:
    """
    Represents a single user with their preferences, taste profile, and dual embeddings.
    """
    
    def __init__(self, name: str, preferences: UserPreferences = None, special: bool = False):
        """
        Initialize a new user.
        
        Args:
            name: User's name
            preferences: UserPreferences object containing books, movies, music
            special: Whether this user is marked as special (default: False)
        """
        self.name = name
        self.preferences = preferences or UserPreferences(books=[])
        
        # Dual profile system
        self.interests_profile = None  # 200-word interests summary
        self.personality_profile = None  # Dict of 5 personality trait descriptions
        
        # Dual embedding system
        self.interests_embedding = None  # 3072-dimensional vector
        self.personality_embedding = None  # 3840-dimensional vector (5 x 768)
        
        # Legacy fields for backward compatibility
        self.taste_profile = None
        self.embedding = None
        
        # Additional data from frontend JSON
        self.profile_data = {}  # Basic profile info
        self.personality_responses = {}  # Big 5 responses
        self.additional_info = {}  # Extra questions
        
        self.special = special
        self.created_at = datetime.now()
        self.metadata = {}
    
    def add_book(self, title: str, author: str, rating: float, genre: str = None):
        """Add a book rating to the user's preferences."""
        book = BookRating(title=title, author=author, rating=rating, genre=genre)
        self.preferences.books.append(book)
    
    def add_movie(self, title: str, rating: float, **kwargs):
        """Add a movie rating to the user's preferences."""
        movie = {"title": title, "rating": rating, **kwargs}
        self.preferences.movies.append(movie)
    
    def add_music(self, artist: str, rating: float, **kwargs):
        """Add a music preference to the user's collection."""
        music = {"artist": artist, "rating": rating, **kwargs}
        self.preferences.music.append(music)
    
    def set_taste_profile(self, profile: str):
        """Set the AI-generated taste profile description."""
        self.taste_profile = profile
    
    def set_embedding(self, embedding: np.ndarray):
        """Set the vector embedding for this user."""
        self.embedding = embedding.copy() if isinstance(embedding, np.ndarray) else np.array(embedding)
    
    def set_interests_profile(self, profile: str):
        """Set the AI-generated interests profile description."""
        self.interests_profile = profile
    
    def set_personality_profile(self, profiles: Dict[str, str]):
        """Set the AI-generated personality trait descriptions."""
        self.personality_profile = profiles.copy()
    
    def set_interests_embedding(self, embedding: np.ndarray):
        """Set the 3072-dimensional interests embedding."""
        self.interests_embedding = embedding.copy() if isinstance(embedding, np.ndarray) else np.array(embedding)
    
    def set_personality_embedding(self, embedding: np.ndarray):
        """Set the 3840-dimensional personality embedding."""
        self.personality_embedding = embedding.copy() if isinstance(embedding, np.ndarray) else np.array(embedding)
    
    def get_books_summary(self) -> str:
        """Get a formatted summary of the user's book preferences."""
        if not self.preferences.books:
            return "No books rated yet."
        
        summary = f"{self.name}'s book preferences:\n"
        for book in self.preferences.books:
            summary += f"- {book.title} by {book.author}: {book.rating}/5 stars\n"
        return summary
    
    def get_high_rated_books(self, min_rating: float = 4.0) -> List[BookRating]:
        """Get books rated above a certain threshold."""
        return [book for book in self.preferences.books if book.rating >= min_rating]
    
    @classmethod
    def from_frontend_json(cls, json_data: Dict[str, Any]) -> 'User':
        """
        Create a User instance from frontend JSON format.
        
        Args:
            json_data: Dictionary in the format exported from the React frontend
            
        Returns:
            User instance with populated data
        """
        # Extract basic profile data
        profile = json_data.get("profile", {})
        name = profile.get("fullName", "Unknown User")
        
        # Create user instance
        user = cls(name=name)
        
        # Store profile data
        user.profile_data = profile
        
        # Store personality responses
        user.personality_responses = json_data.get("personality", {})
        
        # Store additional info
        user.additional_info = json_data.get("additionalInfo", {})
        
        # Extract media preferences
        books_data = json_data.get("books", {})
        movies_data = json_data.get("movies", {})
        music_data = json_data.get("music", {})
        
        # Convert to internal format
        user.preferences = UserPreferences(books=[])
        
        # Add books
        for book_title in books_data.get("favoriteBooks", []):
            # Get review if available
            review = books_data.get("bookReviews", {}).get(book_title, "")
            # Create book entry (we don't have author/rating in this format)
            user.add_book(book_title, "Unknown Author", 5.0, "Unknown")
        
        # Add movies
        for movie_title in movies_data.get("favoriteMovies", []):
            review = movies_data.get("movieReviews", {}).get(movie_title, "")
            user.add_movie(movie_title, 5.0, review=review)
        
        # Add music
        for artist in music_data.get("musicArtists", []):
            user.add_music(artist, 5.0)
        
        # Store metadata including original frontend data
        user.metadata = {
            "source": "frontend_json",
            "timestamp": json_data.get("metadata", {}).get("timestamp"),
            "version": json_data.get("metadata", {}).get("version", "1.0"),
            "frontend_data": json_data  # Store complete original data
        }
        
        return user
    
    def calculate_similarity(self, other_user: 'User', mode: str = 'combined') -> float:
        """
        Calculate cosine similarity with another user based on embeddings.
        
        Args:
            other_user: Another User instance
            mode: 'combined', 'interests', 'personality', or 'legacy'
            
        Returns:
            Cosine similarity score (0-1)
        """
        if mode == 'interests':
            if self.interests_embedding is None or other_user.interests_embedding is None:
                raise ValueError("Both users must have interests embeddings")
            vec1, vec2 = self.interests_embedding, other_user.interests_embedding
            
        elif mode == 'personality':
            if self.personality_embedding is None or other_user.personality_embedding is None:
                raise ValueError("Both users must have personality embeddings")
            vec1, vec2 = self.personality_embedding, other_user.personality_embedding
            
        elif mode == 'combined':
            if (self.interests_embedding is None or self.personality_embedding is None or 
                other_user.interests_embedding is None or other_user.personality_embedding is None):
                raise ValueError("Both users must have both embedding types for combined similarity")
            
            # Concatenate interests and personality embeddings
            vec1 = np.concatenate([self.interests_embedding, self.personality_embedding])
            vec2 = np.concatenate([other_user.interests_embedding, other_user.personality_embedding])
            
        else:  # legacy mode
            if self.embedding is None or other_user.embedding is None:
                raise ValueError("Both users must have legacy embeddings")
            vec1, vec2 = self.embedding, other_user.embedding
        
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(similarity)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary format."""
        return {
            "name": self.name,
            "special": self.special,
            "preferences": {
                "books": [asdict(book) for book in self.preferences.books],
                "movies": self.preferences.movies,
                "music": self.preferences.music
            },
            # Dual profile system
            "interests_profile": self.interests_profile,
            "personality_profile": self.personality_profile,
            "interests_embedding": self.interests_embedding.tolist() if self.interests_embedding is not None else None,
            "personality_embedding": self.personality_embedding.tolist() if self.personality_embedding is not None else None,
            
            # Legacy fields (only include if they exist)
            "taste_profile": getattr(self, 'taste_profile', None),
            "embedding": getattr(self, 'embedding', None).tolist() if hasattr(self, 'embedding') and self.embedding is not None else None,
            
            # Frontend data
            "profile_data": self.profile_data,
            "personality_responses": self.personality_responses,
            "additional_info": self.additional_info,
            
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create a User instance from dictionary data."""
        # Create preferences
        books = [BookRating(**book_data) for book_data in data.get("preferences", {}).get("books", [])]
        preferences = UserPreferences(
            books=books,
            movies=data.get("preferences", {}).get("movies", []),
            music=data.get("preferences", {}).get("music", [])
        )
        
        # Create user with special field
        user = cls(
            name=data["name"], 
            preferences=preferences,
            special=data.get("special", False)
        )
        
        # Load dual profile system
        user.interests_profile = data.get("interests_profile")
        user.personality_profile = data.get("personality_profile")
        
        if data.get("interests_embedding"):
            user.interests_embedding = np.array(data["interests_embedding"])
        
        if data.get("personality_embedding"):
            user.personality_embedding = np.array(data["personality_embedding"])
        
        # Load legacy fields
        user.taste_profile = data.get("taste_profile")
        if data.get("embedding"):
            user.embedding = np.array(data["embedding"])
        
        # Load frontend data
        user.profile_data = data.get("profile_data", {})
        user.personality_responses = data.get("personality_responses", {})
        user.additional_info = data.get("additional_info", {})
        
        if data.get("created_at"):
            user.created_at = datetime.fromisoformat(data["created_at"])
        
        user.metadata = data.get("metadata", {})
        
        return user
    
    def save_to_json(self, filepath: str):
        """Save user to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'User':
        """Load user from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __str__(self) -> str:
        """String representation of the user."""
        book_count = len(self.preferences.books)
        has_interests_profile = "Yes" if self.interests_profile else "No"
        has_personality_profile = "Yes" if self.personality_profile else "No"
        has_interests_embedding = "Yes" if self.interests_embedding is not None else "No"
        has_personality_embedding = "Yes" if self.personality_embedding is not None else "No"
        special_marker = "⭐" if self.special else ""
        
        return (f"User(name='{self.name}', books={book_count}, "
                f"interests_profile={has_interests_profile}, personality_profile={has_personality_profile}, "
                f"interests_emb={has_interests_embedding}, personality_emb={has_personality_embedding}, "
                f"special={self.special}){special_marker}")
    
    def __repr__(self) -> str:
        return self.__str__()