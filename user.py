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
    Represents a single user with their preferences, taste profile, and embeddings.
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
        self.taste_profile = None
        self.embedding = None
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
    
    def calculate_similarity(self, other_user: 'User') -> float:
        """
        Calculate cosine similarity with another user based on embeddings.
        
        Args:
            other_user: Another User instance
            
        Returns:
            Cosine similarity score (0-1)
        """
        if self.embedding is None or other_user.embedding is None:
            raise ValueError("Both users must have embeddings to calculate similarity")
        
        # Normalize vectors
        norm_self = np.linalg.norm(self.embedding)
        norm_other = np.linalg.norm(other_user.embedding)
        
        if norm_self == 0 or norm_other == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(self.embedding, other_user.embedding) / (norm_self * norm_other)
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
            "taste_profile": self.taste_profile,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
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
        user.taste_profile = data.get("taste_profile")
        
        if data.get("embedding"):
            user.embedding = np.array(data["embedding"])
        
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
        has_profile = "Yes" if self.taste_profile else "No"
        has_embedding = "Yes" if self.embedding is not None else "No"
        special_marker = "⭐" if self.special else ""
        
        return (f"User(name='{self.name}', books={book_count}, "
                f"taste_profile={has_profile}, embedding={has_embedding}, special={self.special}){special_marker}")
    
    def __repr__(self) -> str:
        return self.__str__()