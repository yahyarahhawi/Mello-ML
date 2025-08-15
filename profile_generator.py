#!/usr/bin/env python3
"""
ProfileGenerator class for generating AI-powered taste profiles and user data.
"""

import os
import time
import random
import requests
import json
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False

from user import User, UserPreferences, BookRating
from population import Population


class ProfileGenerator:
    """
    Generates user data and taste profiles using various AI models.
    Handles both synthetic user creation and taste profile generation.
    """
    
    def __init__(self, openrouter_api_key: str = None):
        """
        Initialize the profile generator.
        
        Args:
            openrouter_api_key: OpenRouter API key for LLM calls
        """
        load_dotenv()
        
        self.openrouter_api_key = openrouter_api_key or os.getenv('OPENROUTER_API_KEY')
        
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
        
        self.logger = logging.getLogger(__name__)
        self.request_count = 0
        
        # Initialize Faker for name generation
        if FAKER_AVAILABLE:
            self.fake = Faker()
        else:
            self.logger.warning("Faker not available. Install with: pip install faker")
            self.fake = None
        
        # Model configurations from environment variables
        self.user_generation_model = os.getenv('USER_GENERATION_MODEL', 'moonshotai/kimi-k2:free')
        self.taste_profile_model = os.getenv('TASTE_PROFILE_MODEL', 'google/gemini-2.5-flash')
        
        # Personality archetypes for diverse user generation (50 types)
        self.personality_archetypes = [
    "Academic philosopher who loves ancient wisdom and existential questions",
    "Tech entrepreneur fascinated by science fiction and innovation",
    "Environmental activist drawn to nature writing and climate literature", 
    "History buff obsessed with biographical accounts and war narratives",
    "Psychology student interested in human behavior and self-help",
    "Creative writer who devours literary fiction and poetry",
    "Social justice advocate reading about inequality and activism",
    "Spiritual seeker exploring religious texts and meditation guides",
    "Business professional focused on leadership and strategy books",
    "Art lover reading about creativity, design, and aesthetics",
    "Travel enthusiast who reads memoirs and cultural explorations",
    "Science nerd passionate about physics, biology, and research",
    "Mystery lover addicted to crime fiction and psychological thrillers",
    "Romance reader who enjoys contemporary and historical love stories",
    "Fantasy escapist who lives in magical worlds and epic adventures",
    "Political junkie reading about governance, policy, and current events",
    "Health enthusiast focused on nutrition, fitness, and wellness",
    "Educator interested in learning theory and child development",
    "Musician who reads about music theory, biographies, and culture",
    "Chef passionate about food culture, recipes, and culinary history",
    "Minimalist drawn to simplicity, productivity, and intentional living",
    "Social media influencer reading about marketing and personal branding",
    "Retired teacher who enjoys cozy mysteries and gentle fiction",
    "College student exploring identity through coming-of-age stories",
    "Working parent seeking practical advice and quick fiction escapes",
    "Immigrant reader connecting with stories of displacement and belonging",
    "Small town resident who loves community-focused and rural narratives",
    "Urban professional reading about city life and career advancement",
    "Recovering addict finding strength in memoirs and recovery literature",
    "New parent reading about child-rearing and family dynamics",
    "Chronic illness warrior seeking medical narratives and inspiration",
    "LGBTQ+ reader exploring queer literature and identity stories",
    "Military veteran interested in war stories and brotherhood tales",
    "Religious conservative who prefers faith-based and moral literature",
    "Liberal progressive reading about social change and activism",
    "Introvert who loves quiet character studies and introspective novels",
    "Extrovert drawn to adventure stories and social narratives",
    "Anxious person seeking comfort in gentle, predictable stories",
    "Risk-taker who enjoys intense, challenging, and provocative books",
    "Nostalgic reader who gravitates toward historical and vintage settings",
    "Futurist fascinated by dystopian and speculative fiction",
    "Empath who connects deeply with emotional and relationship-focused books",
    "Intellectual who enjoys complex, philosophical, and theoretical works",
    "Practical person who prefers how-to guides and actionable advice",
    "Dreamer who loves magical realism and imaginative storytelling",
    "Skeptic drawn to investigative journalism and fact-based narratives",
    "Optimist who seeks uplifting, inspirational, and hopeful stories",
    "Pessimist attracted to dark, realistic, and challenging literature",
    "Curious generalist who reads widely across all genres and topics",
    "Specialist who deep-dives into one particular subject or genre"
]
    
    def _make_api_call(self, prompt: str, model: str, max_tokens: int = 200, temperature: float = 0.7) -> Optional[str]:
        """
        Make API call to OpenRouter.
        
        Args:
            prompt: Text prompt
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text or None if failed
        """
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            # Add small delay to avoid rate limits
            time.sleep(random.uniform(0.5, 1.5))
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            self.request_count += 1
            
            return result['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            self.logger.error(f"API call error: {e}")
            return None
    
    def _parse_book_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the structured text response into book data.
        
        Args:
            response: Text response from LLM
            
        Returns:
            List of book dictionaries
        """
        books = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or not line.startswith('Title:'):
                continue
            
            try:
                # Parse the line format: Title: [title] | Author: [author] | Rating: [rating] | Genre: [genre]
                parts = line.split(' | ')
                
                if len(parts) >= 4:
                    title = parts[0].replace('Title:', '').strip()
                    author = parts[1].replace('Author:', '').strip()
                    rating_str = parts[2].replace('Rating:', '').strip()
                    genre = parts[3].replace('Genre:', '').strip()
                    
                    # Convert rating to float
                    try:
                        rating = float(rating_str)
                        rating = max(1.0, min(5.0, rating))  # Clamp between 1-5
                    except ValueError:
                        rating = 3.0  # Default rating
                    
                    books.append({
                        'title': title,
                        'author': author,
                        'rating': rating,
                        'genre': genre
                    })
            
            except Exception as e:
                self.logger.warning(f"Failed to parse line: {line[:50]}... Error: {e}")
                continue
        
        # If we didn't get enough books, add some defaults
        while len(books) < 20:
            books.append({
                'title': f'Default Book {len(books) + 1}',
                'author': 'Unknown Author',
                'rating': 3.0,
                'genre': 'Fiction'
            })
        
        return books[:20]  # Ensure exactly 20 books
    
    def generate_synthetic_user(self, personality_type: str = None) -> Optional[User]:
        """
        Generate a synthetic user with book preferences.
        
        Args:
            personality_type: Specific personality type, or random if None
            
        Returns:
            User object with generated preferences
        """
        if personality_type is None:
            personality_type = random.choice(self.personality_archetypes)
        
        # Generate realistic name using Faker
        if self.fake:
            name = self.fake.name()
        else:
            # Fallback if Faker not available
            name = f"User_{random.randint(1000, 9999)}"
        
        # Create prompt for user generation (without name generation)
        prompt = f"""Generate exactly 20 books that a person with personality type "{personality_type}" would rate.

For each book, provide:
- Title
- Author  
- Rating (1.0 to 5.0)
- Genre

Include a mix of fiction and non-fiction (philosophy, self-help, religion, science, etc.).
Make ratings reflect their personality.

Format each book on a new line as:
Title: [book title] | Author: [author name] | Rating: [1.0-5.0] | Genre: [genre]

Example:
Title: The Bell Jar | Author: Sylvia Plath | Rating: 4.5 | Genre: Fiction
Title: Sapiens | Author: Yuval Noah Harari | Rating: 4.0 | Genre: Non-fiction

Generate 20 books total."""
        
        response = self._make_api_call(prompt, self.user_generation_model, max_tokens=800, temperature=0.8)
        
        if not response:
            return None
        
        try:
            # Parse the structured text response
            books_data = self._parse_book_response(response)
            
            # Create book ratings
            books = []
            for book_data in books_data:
                book = BookRating(
                    title=book_data.get('title', 'Unknown Title'),
                    author=book_data.get('author', 'Unknown Author'),
                    rating=float(book_data.get('rating', 3.0)),
                    genre=book_data.get('genre', 'Unknown')
                )
                books.append(book)
            
            # Create preferences
            preferences = UserPreferences(books=books)
            
            # Create user
            user = User(name=name, preferences=preferences)
            user.metadata['personality_type'] = personality_type
            user.metadata['generated'] = True
            
            self.logger.info(f"Generated synthetic user: {name} ({personality_type})")
            return user
            
        except Exception as e:
            self.logger.error(f"Error creating user: {e}")
            self.logger.error(f"Response was: {response[:200]}...")
            return None
    
    def generate_taste_profile(self, user: User) -> bool:
        """
        Generate an AI taste profile for a user based on their book preferences.
        
        Args:
            user: User to generate profile for
            
        Returns:
            True if successful, False otherwise
        """
        if not user.preferences.books:
            self.logger.error(f"User {user.name} has no books to analyze")
            return False
        
        # Create book summary
        book_summary = f"{user.name}'s book preferences:\n"
        for book in user.preferences.books:
            book_summary += f"- {book.title} by {book.author}: {book.rating}/5 stars ({book.genre})\n"
        
        # Create prompt for taste profile generation
        prompt = f""" {book_summary}
        Based on the following list of books, movies, and personality question responses, write a 200-word personality profile that captures how this person thinks, feels, and behaves.
        • Use “they” as the pronoun. do not mention name
        • Focus on inferring personality traits from the combined inputs, including cognitive style, emotional tendencies, values, motivations, and social orientation.
        • Use clear, semantically rich language with personality-relevant descriptors (e.g., analytical, empathetic, adventurous, introverted, pragmatic, idealistic, resilient).
        • Avoid listing titles, authors, or specific movie names.
        • Avoid literary analysis; instead, describe character attributes, worldview, and patterns of thought.
        • Organize the profile into a cohesive narrative rather than bullet points.
        • Avoid overly poetic language or abstract metaphors — keep it concrete, direct, and precise.
        • Output exactly one paragraph in plain text, without headings or formatting.
        • The description should be well-rounded, covering intellectual interests, emotional landscape, interpersonal style, and decision-making tendencies.

        Do not rely on one book or one review heavily. make your analysis general
        """
        
        response = self._make_api_call(prompt, self.taste_profile_model, max_tokens=150, temperature=1.2)
        
        if response:
            user.set_taste_profile(response)
            self.logger.info(f"Generated taste profile for {user.name}")
            return True
        else:
            self.logger.error(f"Failed to generate taste profile for {user.name}")
            return False
    
    def generate_population(self, size: int = 50, progress_callback=None) -> Population:
        """
        Generate a population of synthetic users.
        
        Args:
            size: Number of users to generate
            progress_callback: Function to call with progress updates
            
        Returns:
            Population with generated users
        """
        population = Population(f"Generated Population ({size} users)")
        
        self.logger.info(f"Generating {size} synthetic users...")
        
        # Ensure diverse personality types
        personality_cycle = (self.personality_archetypes * ((size // len(self.personality_archetypes)) + 1))[:size]
        random.shuffle(personality_cycle)
        
        successful_generations = 0
        
        for i in range(size):
            personality_type = personality_cycle[i]
            user = self.generate_synthetic_user(personality_type)
            
            if user:
                population.add_user(user)
                successful_generations += 1
            
            if progress_callback:
                progress_callback(i + 1, size)
            
            # Log progress
            if (i + 1) % 10 == 0:
                self.logger.info(f"Generated {successful_generations}/{i + 1} users")
        
        self.logger.info(f"Successfully generated {successful_generations}/{size} users")
        return population
    
    def generate_taste_profiles_for_population(self, population: Population, progress_callback=None) -> int:
        """
        Generate taste profiles for all users in a population.
        
        Args:
            population: Population to generate profiles for
            progress_callback: Function to call with progress updates
            
        Returns:
            Number of successfully generated profiles
        """
        users_without_profiles = [user for user in population.users if not user.taste_profile]
        
        if not users_without_profiles:
            self.logger.info("All users already have taste profiles")
            return 0
        
        self.logger.info(f"Generating taste profiles for {len(users_without_profiles)} users")
        
        successful_profiles = 0
        
        for i, user in enumerate(users_without_profiles):
            if self.generate_taste_profile(user):
                successful_profiles += 1
            
            if progress_callback:
                progress_callback(i + 1, len(users_without_profiles))
        
        self.logger.info(f"Successfully generated {successful_profiles}/{len(users_without_profiles)} taste profiles")
        return successful_profiles
    
    def create_user_from_books(self, name: str, books_data: List[Dict[str, Any]], special: bool = False) -> User:
        """
        Create a user from a list of book data.
        
        Args:
            name: User's name
            books_data: List of dicts with 'title', 'author', 'rating', 'genre'
            special: Whether to mark this user as special
            
        Returns:
            User object
        """
        books = []
        for book_data in books_data:
            book = BookRating(
                title=book_data.get('title', 'Unknown Title'),
                author=book_data.get('author', 'Unknown Author'),
                rating=float(book_data.get('rating', 3.0)),
                genre=book_data.get('genre', 'Unknown')
            )
            books.append(book)
        
        preferences = UserPreferences(books=books)
        user = User(name=name, preferences=preferences, special=special)
        user.metadata['manually_created'] = True
        
        return user
    
    def load_legacy_population(self, filepath: str) -> Population:
        """
        Load population from legacy JSON format.
        
        Args:
            filepath: Path to legacy JSON file
            
        Returns:
            Population object
        """
        with open(filepath, 'r') as f:
            legacy_data = json.load(f)
        
        population = Population("Legacy Population")
        
        for user_data in legacy_data:
            if isinstance(user_data, dict) and 'name' in user_data:
                user = User(name=user_data['name'])
                
                # Set taste profile if available
                if 'book_taste' in user_data:
                    user.set_taste_profile(user_data['book_taste'])
                
                # Set embedding if available
                if 'books_vector' in user_data:
                    user.set_embedding(user_data['books_vector'])
                
                population.add_user(user)
        
        self.logger.info(f"Loaded {len(population.users)} users from legacy file")
        return population
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about profile generation."""
        return {
            "user_generation_model": self.user_generation_model,
            "taste_profile_model": self.taste_profile_model,
            "total_requests": self.request_count,
            "personality_archetypes": len(self.personality_archetypes)
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"ProfileGenerator(requests={self.request_count}, archetypes={len(self.personality_archetypes)})"