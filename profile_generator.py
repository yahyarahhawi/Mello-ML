#!/usr/bin/env python3
"""
Fresh ProfileGenerator with unified personality approach and 50 archetypes.
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

from user import User


class ProfileGenerator:
    """
    Generates unified personality profiles from cultural data and Big 5 responses.
    Uses 50 archetypes for diverse synthetic user generation.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the profile generator.
        
        Args:
            api_key: OpenRouter API key
        """
        load_dotenv()
        
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
        
        self.logger = logging.getLogger(__name__)
        self.request_count = 0
        
        # Initialize Faker
        if FAKER_AVAILABLE:
            self.fake = Faker()
        else:
            self.logger.warning("Faker not available. Install with: pip install faker")
            self.fake = None
        
        # Model configuration from .env
        self.model = os.getenv('TASTE_PROFILE_MODEL', 'google/gemini-2.5-flash')
        self.user_generation_model = os.getenv('USER_GENERATION_MODEL', 'google/gemini-2.5-flash')
        
        # 50 personality archetypes for diverse generation
        self.archetypes = [
            "bookworm introvert who loves classic literature and philosophical works",
            "creative arts student into experimental films and indie music",
            "sci-fi enthusiast fascinated by space, technology, and dystopian futures",
            "pop culture fanatic who follows mainstream trends and blockbuster movies",
            "minimalist who prefers simple, profound stories and acoustic music",
            "history buff interested in historical fiction and period dramas",
            "psychology student drawn to character studies and complex narratives",
            "adventure seeker who loves travel memoirs and action movies",
            "romantic optimist who enjoys love stories and feel-good entertainment",
            "social activist interested in diverse voices and social justice themes",
            "mystery lover obsessed with crime novels and thriller films",
            "fantasy escapist who prefers magical worlds and epic soundtracks",
            "comedy enthusiast who gravitates toward humor and satire",
            "nature lover interested in environmental themes and outdoor adventures",
            "urban dweller fascinated by city life and contemporary culture",
            "retro enthusiast who loves vintage books, classic films, and old music",
            "international student with diverse cultural tastes",
            "sports fan who enjoys biographies and sports movies",
            "entrepreneur interested in business books and motivational content",
            "art student passionate about visual storytelling and experimental media",
            "tech geek drawn to cyberpunk and innovation narratives",
            "spiritual seeker exploring meditation, philosophy, and consciousness",
            "rebel nonconformist attracted to counterculture and underground art",
            "academic perfectionist focused on intellectual rigor and classical works",
            "social butterfly who loves celebrity culture and trending content",
            "nostalgic traditionalist preferring timeless stories and familiar genres",
            "curious experimenter always trying new artistic forms and genres",
            "practical realist who prefers grounded, realistic narratives",
            "dreamy idealist drawn to utopian visions and inspirational content",
            "dark aesthete fascinated by gothic, noir, and psychological themes",
            "optimistic humanist interested in uplifting stories about human potential",
            "analytical thinker who enjoys puzzles, mysteries, and logical narratives",
            "emotional empath drawn to heart-wrenching dramas and personal stories",
            "adrenaline junkie who seeks intense, high-energy entertainment",
            "peaceful mediator preferring harmonious, conflict-free content",
            "ambitious climber interested in success stories and power dynamics",
            "free spirit attracted to bohemian culture and artistic expression",
            "methodical planner who enjoys structured narratives and clear resolutions",
            "spontaneous adventurer open to random discoveries and surprises",
            "introspective philosopher contemplating existence and meaning",
            "extroverted performer drawn to theatrical and dramatic content",
            "quiet observer interested in subtle, understated artistic expression",
            "passionate activist engaged with political and social commentary",
            "escapist dreamer seeking fantasy worlds and imaginative stories",
            "grounded pragmatist preferring practical, applicable content",
            "sensitive artist attracted to beauty, aesthetics, and emotional depth",
            "logical scientist interested in factual, evidence-based narratives",
            "cultural curator who appreciates diverse artistic traditions",
            "trend follower who stays current with popular cultural movements",
            "timeless classicist who values enduring artistic achievements"
        ]
        
        # Big 5 trait options (same as original system)
        self.trait_options = {
            'Openness': [
                'I seek out restaurants serving cuisine I\'ve never tried',
                'I regularly explore music outside my usual genres', 
                'I stick mostly to familiar authors or genres when reading',
                'I prefer detailed itineraries when traveling somewhere new',
                'I enjoy unusual or experimental art even if I don\'t fully understand it',
                'I use the same apps and rarely try new ones',
                'I shop at the same stores and buy similar clothes repeatedly',
                'I prefer exploring new neighborhoods over returning to familiar ones',
                'I enjoy learning about unfamiliar topics just for fun',
                'I rewatch the same TV shows or movies for comfort',
                'I experiment with new recipes instead of cooking the same meals',
                'I like debating ideas I disagree with to understand other perspectives'
            ],
            'Conscientiousness': [
                'I write down tasks as soon as I think of them',
                'I make my bed without thinking about it',
                'I pack for trips well in advance', 
                'I rarely miss deadlines for assignments or work',
                'I have dozens of unread notifications on my phone',
                'I sometimes run a few minutes late to events',
                'I tidy up my space soon after it gets messy',
                'I track expenses and stick to budgets',
                'I use to-do lists and use them to guide my day',
                'I charge devices before they get close to empty',
                'I leave emails unread for long periods',
                'I set multiple alarms to make sure I wake up',
                'I prepare for exams or big projects weeks in advance',
                'I sometimes lose momentum on personal goals after the initial excitement'
            ],
            'Extraversion': [
                'I volunteer to speak or present in front of groups',
                'At parties, I gravitate toward smaller conversations',
                'I call friends when I have news instead of texting',
                'I often eat or take breaks alone even when others are around',
                'Large social events leave me tired afterward',
                'I think out loud when working through ideas',
                'I regularly invite others to hang out',
                'I sometimes start conversations with strangers',
                'I enjoy solo hobbies like reading or gaming',
                'I prefer quiet study spaces over busy ones',
                'I feel energized after spending time with groups',
                'I need alone time to recharge after socializing',
                'I share personal updates only when asked',
                'I often get "in the zone" when working alone'
            ],
            'Agreeableness': [
                'I split bills evenly, even if I ordered less or more',
                'I\'m fine letting others choose the music or activity',
                'I tend to stay on the phone until the other person hangs up',
                'I offer help without waiting to be asked',
                'I sometimes push for my choice in group decisions',
                'I wait to trust people until I know them better',
                'I check what everyone wants before ordering for a group',
                'I apologize when I realize I hurt someone',
                'I remember birthdays or important events for people close to me',
                'I tell people directly when something bothers me',
                'I adjust my plans to fit others\' schedules',
                'I freely lend personal items to friends',
                'I speak up when someone is treated unfairly',
                'I sometimes put my own needs ahead of others\' requests'
            ],
            'Neuroticism': [
                'I reread emails before sending them to avoid mistakes',
                'I sometimes lose sleep over upcoming events or deadlines',
                'I get annoyed when unexpected delays happen',
                'I rarely dwell on things I can\'t control',
                'My mood changes easily depending on what\'s happening',
                'I\'m usually calm before social events',
                'I sometimes replay conversations or moments in my head',
                'I make backup plans in case the first one fails',
                'I occasionally worry about embarrassing myself',
                'Small frustrations can affect my mood for hours',
                'I recover quickly after disappointment',
                'I can laugh off small mistakes easily',
                'I stay focused in emergencies or high-pressure situations',
                'I sometimes second-guess decisions I\'ve already made'
            ]
        }
    
    def _make_api_call(self, prompt: str, max_tokens: int = 300, temperature: float = 0.7, model: str = None) -> Optional[str]:
        """Make API call to OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model or self.model,
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
    
    def generate_synthetic_user_data(self, archetype: str = None) -> Optional[Dict[str, Any]]:
        """
        Generate complete synthetic user data following real user JSON format.
        
        Args:
            archetype: Personality archetype, or random if None
            
        Returns:
            Complete user data in JSON format
        """
        if archetype is None:
            archetype = random.choice(self.archetypes)
        
        # Generate name
        if self.fake:
            name = self.fake.name()
        else:
            name = f"User_{random.randint(1000, 9999)}"
        
        self.logger.info(f"Generating synthetic user: {name} ({archetype})")
        
        # Generate profile section
        profile_prompt = f"""Generate profile data for a college student who is: {archetype}
        
        Format exactly:
        fullName: {name}
        classYear: [2024, 2025, 2026, or 2027]
        major: [realistic major that fits their personality]
        bio: [2-3 sentences in their authentic voice]
        interests: [5-7 interests as comma-separated list]"""
        
        profile_response = self._make_api_call(profile_prompt, max_tokens=200, temperature=1.0, model=self.user_generation_model)
        if not profile_response:
            return None
        
        # Parse profile
        profile_data = {"fullName": name}
        for line in profile_response.split('\\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                if key == 'interests':
                    profile_data[key] = [i.strip() for i in value.split(',')]
                else:
                    profile_data[key] = value
        
        # Generate books section
        books_prompt = f"""For someone who is: {archetype}
        
        Generate:
        favoriteBooks: [5 real book titles that fit their taste]
        bookReviews: [3 reviews in their voice, 1-2 sentences each]
        bookReflection: [why they choose these books, 1-2 sentences]
        
        Format as JSON object."""
        
        books_response = self._make_api_call(books_prompt, max_tokens=400, temperature=1.0, model=self.user_generation_model)
        books_data = self._parse_json_response(books_response)
        if not isinstance(books_data, dict):
            books_data = {}
        
        # Generate movies section
        movies_prompt = f"""For someone who is: {archetype}
        
        Generate:
        favoriteMovies: [5 real movie/show titles that fit their taste]
        movieReviews: [3 reviews in their voice, 1-2 sentences each]
        movieReflection: [why they're drawn to these films, 1-2 sentences]
        
        Format as JSON object."""
        
        movies_response = self._make_api_call(movies_prompt, max_tokens=400, temperature=1.0, model=self.user_generation_model)
        movies_data = self._parse_json_response(movies_response)
        if not isinstance(movies_data, dict):
            movies_data = {}
        
        # Generate music section
        music_prompt = f"""For someone who is: {archetype}
        
        Generate:
        musicArtists: [5 real artists/bands that fit their taste]
        vibeMatch: [how they'd describe their music taste, 1-2 sentences]
        
        Format as JSON object."""
        
        music_response = self._make_api_call(music_prompt, max_tokens=200, temperature=1.0, model=self.user_generation_model)
        music_data = self._parse_json_response(music_response)
        if not isinstance(music_data, dict):
            music_data = {}
        
        # Generate personality responses
        personality_data = {}
        for trait in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']:
            # Determine trait level from archetype
            trait_level = self._infer_trait_level(archetype, trait)
            
            # Select 4-6 behaviors that match the trait level
            all_options = self.trait_options[trait]
            if trait_level == 'high':
                # Select more positive behaviors
                selected = random.sample(all_options[:8], random.randint(4, 6))
            elif trait_level == 'low':
                # Select more limiting behaviors
                selected = random.sample(all_options[4:], random.randint(4, 6))
            else:  # moderate
                # Mix of both
                selected = random.sample(all_options, random.randint(4, 6))
            
            not_selected = [opt for opt in all_options if opt not in selected]
            
            personality_data[trait] = {
                'selected': selected,
                'not_selected': not_selected,
                'notes': f"This reflects my {trait_level} {trait.lower()}."
            }
        
        # Generate additional info
        additional_prompt = f"""For someone who is: {archetype}
        
        Create a JSON object with these fields (short responses):
        talkAboutForHours: [one thing they could discuss endlessly, 1 sentence]
        perfectDay: [describe their ideal day, 1-2 sentences]
        energySources: [list 3 things that energize them]
        
        Format as valid JSON only."""
        
        additional_response = self._make_api_call(additional_prompt, max_tokens=250, temperature=1.0, model=self.user_generation_model)
        additional_data = self._parse_json_response(additional_response)
        if not isinstance(additional_data, dict):
            additional_data = {}
        
        # Assemble complete user data
        user_data = {
            "profile": profile_data,
            "books": books_data,
            "movies": movies_data,
            "music": music_data,
            "personality": personality_data,
            "additionalInfo": additional_data,
            "metadata": {
                "archetype": archetype,
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "2.0"
            }
        }
        
        return user_data
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from LLM response."""
        if not response:
            return None
        
        try:
            # Remove markdown code block markers if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]   # Remove ```
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove trailing ```
            
            # Try to find JSON in the response
            start = cleaned_response.find('{')
            end = cleaned_response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = cleaned_response[start:end]
                return json.loads(json_str)
        except Exception as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            pass
        
        return None
    
    def _infer_trait_level(self, archetype: str, trait: str) -> str:
        """Infer trait level from archetype description."""
        archetype_lower = archetype.lower()
        
        # Simple heuristics based on archetype keywords
        if trait == 'Openness':
            if any(word in archetype_lower for word in ['experimental', 'diverse', 'curious', 'new', 'creative', 'artistic']):
                return 'high'
            elif any(word in archetype_lower for word in ['traditional', 'classic', 'familiar', 'timeless']):
                return 'low'
        elif trait == 'Conscientiousness':
            if any(word in archetype_lower for word in ['perfectionist', 'academic', 'methodical', 'planner']):
                return 'high'
            elif any(word in archetype_lower for word in ['spontaneous', 'free spirit', 'rebel']):
                return 'low'
        elif trait == 'Extraversion':
            if any(word in archetype_lower for word in ['social', 'performer', 'extroverted', 'butterfly']):
                return 'high'
            elif any(word in archetype_lower for word in ['introvert', 'quiet', 'observer', 'bookworm']):
                return 'low'
        elif trait == 'Agreeableness':
            if any(word in archetype_lower for word in ['activist', 'humanist', 'mediator', 'helper']):
                return 'high'
            elif any(word in archetype_lower for word in ['rebel', 'nonconformist', 'competitive']):
                return 'low'
        elif trait == 'Neuroticism':
            if any(word in archetype_lower for word in ['sensitive', 'anxious', 'emotional']):
                return 'high'
            elif any(word in archetype_lower for word in ['calm', 'stable', 'optimist', 'confident']):
                return 'low'
        
        return 'moderate'  # Default
    
    def generate_interests_profile(self, user: User) -> bool:
        """
        Generate unified personality profile from cultural data (legacy approach).
        
        Args:
            user: User to generate profile for
            
        Returns:
            True if successful
        """
        # Build comprehensive input
        input_data = self._build_cultural_input(user)
        
        prompt = f"""Based on "Interests", "books", "movies", "music", "additionalInfo" write a 200-word personality profile that captures how this person thinks, feels, and behaves.
        
        {input_data}
        
        • Use "they" as the pronoun. do not mention name
        • Focus on inferring personality traits from the combined inputs, including cognitive style, emotional tendencies, values, motivations, and social orientation.
        • Use clear, semantically rich language with personality-relevant descriptors (e.g., analytical, empathetic, adventurous, introverted, pragmatic, idealistic, resilient).
        • Avoid listing titles, authors, or specific movie names. Mention genres, themes, high level topics like science, philosophy, morality
        • Avoid literary analysis; instead, describe character attributes, worldview, and patterns of thought.
        • Organize the profile into a cohesive narrative rather than bullet points.
        • Avoid overly poetic language or abstract metaphors — keep it concrete, direct, and precise.
        • Output exactly one paragraph in plain text, without headings or formatting.
        • The description should be well-rounded, covering intellectual interests, emotional landscape, interpersonal style, and decision-making tendencies.
        • Do not rely on one book or one review heavily. make your analysis general"""
        
        response = self._make_api_call(prompt, max_tokens=300, temperature=0.7, model=self.model)
        
        if response:
            user.set_interests_profile(response)
            self.logger.info(f"Generated interests profile for {user.name}")
            return True
        else:
            self.logger.error(f"Failed to generate interests profile for {user.name}")
            return False
    
    def generate_personality_profiles(self, user: User) -> bool:
        """
        Generate individual trait profiles using unified approach.
        
        Args:
            user: User to generate profiles for
            
        Returns:
            True if successful
        """
        # Build input including personality responses
        personality_input = self._build_personality_input(user)
        
        prompt = f"""Based on the personality responses below, generate individual trait descriptions.
        
        {personality_input}
        
        Create semantically rich and descriptive paragraphs for each Big 5 trait. Consider the whole personality for deeper cohesive personality types.
        
        Return JSON object with the 5 traits. Do NOT mention names or pronouns.
        
        {{
          "Openness": "<descriptive paragraph about their openness>",
          "Conscientiousness": "<descriptive paragraph about their conscientiousness>",
          "Extraversion": "<descriptive paragraph about their extraversion>",
          "Agreeableness": "<descriptive paragraph about their agreeableness>",
          "Neuroticism": "<descriptive paragraph about their neuroticism>"
        }}"""
        
        response = self._make_api_call(prompt, max_tokens=800, temperature=0.6, model=self.model)
        
        if response:
            profiles = self._parse_json_response(response)
            if profiles:
                if len(profiles) == 5:
                    user.set_personality_profiles(profiles)
                    self.logger.info(f"Generated personality profiles for {user.name}")
                    return True
                else:
                    self.logger.error(f"Expected 5 personality traits, got {len(profiles)}: {list(profiles.keys()) if profiles else None}")
            else:
                self.logger.error(f"Failed to parse JSON from personality response: {response[:200]}...")
        else:
            self.logger.error(f"No response from API for personality profiles")
        
        self.logger.error(f"Failed to generate personality profiles for {user.name}")
        return False
    
    def _build_cultural_input(self, user: User) -> str:
        """Build cultural input string for interests profile."""
        input_parts = []
        
        # Profile interests
        if user.profile_data.get('interests'):
            input_parts.append(f"Interests: {', '.join(user.profile_data['interests'])}")
        
        # Books
        if user.books_data.get('favoriteBooks'):
            input_parts.append(f"Favorite Books: {', '.join(user.books_data['favoriteBooks'])}")
        
        # Book reviews
        book_reviews = user.books_data.get('bookReviews', {})
        if isinstance(book_reviews, dict):
            for book, review in book_reviews.items():
                input_parts.append(f"Review of {book}: {review}")
        elif isinstance(book_reviews, list):
            for review in book_reviews:
                input_parts.append(f"Book review: {review}")
        
        # Movies
        if user.movies_data.get('favoriteMovies'):
            input_parts.append(f"Favorite Movies: {', '.join(user.movies_data['favoriteMovies'])}")
        
        # Movie reviews
        movie_reviews = user.movies_data.get('movieReviews', {})
        if isinstance(movie_reviews, dict):
            for movie, review in movie_reviews.items():
                input_parts.append(f"Review of {movie}: {review}")
        elif isinstance(movie_reviews, list):
            for review in movie_reviews:
                input_parts.append(f"Movie review: {review}")
        
        # Music
        if user.music_data.get('musicArtists'):
            input_parts.append(f"Music Artists: {', '.join(user.music_data['musicArtists'])}")
        
        if user.music_data.get('vibeMatch'):
            input_parts.append(f"Music Vibe: {user.music_data['vibeMatch']}")
        
        # Additional info
        if isinstance(user.additional_info, dict):
            for key, value in user.additional_info.items():
                if value:
                    input_parts.append(f"{key}: {value}")
        elif isinstance(user.additional_info, list):
            # Handle case where additional_info is a list
            for item in user.additional_info:
                if item:
                    input_parts.append(f"Additional: {item}")
        else:
            # Handle other types
            if user.additional_info:
                input_parts.append(f"Additional: {user.additional_info}")
        
        return '\\n\\n'.join(input_parts)
    
    def _build_personality_input(self, user: User) -> str:
        """Build personality input string."""
        input_parts = []
        
        for trait, data in user.personality_data.items():
            selected = data.get('selected', [])
            not_selected = data.get('not_selected', [])
            notes = data.get('notes', '')
            
            input_parts.append(f"{trait}:")
            input_parts.append(f"  Selected behaviors: {selected}")
            input_parts.append(f"  Not selected: {not_selected}")
            if notes:
                input_parts.append(f"  Notes: {notes}")
            input_parts.append("")
        
        return '\\n'.join(input_parts)
    
    def generate_complete_profiles(self, user: User) -> bool:
        """Generate both interests and personality profiles."""
        interests_success = self.generate_interests_profile(user)
        personality_success = self.generate_personality_profiles(user)
        return interests_success and personality_success
    
    def generate_profile_from_famous_person(self, famous_person_name: str) -> Optional[str]:
        """
        Generate a personality profile based on a famous person.
        
        Args:
            famous_person_name: Name of the famous person to base the profile on
            
        Returns:
            Generated personality profile string, or None if failed
        """
        prompt = f"""Based on what you know about {famous_person_name}, write a 200-word personality profile that captures how this type of person thinks, feels, and behaves.

        • Use "they" as the pronoun. Do not mention the famous person's name directly
        • Focus on personality traits, cognitive style, emotional tendencies, values, motivations, and social orientation
        • Use clear, semantically rich language with personality-relevant descriptors (e.g., analytical, empathetic, adventurous, introverted, pragmatic, idealistic, resilient)
        • Avoid listing specific works, achievements, or biographical details. Focus on general behavioral patterns and personality characteristics
        • Avoid literary analysis; instead, describe character attributes, worldview, and patterns of thought
        • Organize the profile into a cohesive narrative rather than bullet points
        • Avoid overly poetic language or abstract metaphors — keep it concrete, direct, and precise
        • Output exactly one paragraph in plain text, without headings or formatting
        • The description should be well-rounded, covering intellectual interests, emotional landscape, interpersonal style, and decision-making tendencies
        • Focus on the personality type rather than specific historical facts

        Generate a profile that captures the essence of this person's personality type without revealing who they are."""
        
        response = self._make_api_call(prompt, max_tokens=300, temperature=0.7, model=self.model)
        
        if response:
            self.logger.info(f"Generated profile based on {famous_person_name}")
            return response
        else:
            self.logger.error(f"Failed to generate profile for {famous_person_name}")
            return None

    def generate_personality_profiles_from_famous_person(self, famous_person_name: str) -> Optional[Dict[str, str]]:
        """
        Generate Big 5 personality trait profiles based on a famous person.
        
        Args:
            famous_person_name: Name of the famous person to base the profiles on
            
        Returns:
            Dictionary with Big 5 trait descriptions, or None if failed
        """
        prompt = f"""Based on what you know about {famous_person_name}, generate personality descriptions for each of the Big 5 traits. For each trait, write a 2-3 sentence description of how this type of person would manifest that trait.

        The Big 5 traits are:
        - Openness: creativity, curiosity, intellectual engagement
        - Conscientiousness: organization, discipline, goal-orientation  
        - Extraversion: social energy, assertiveness, enthusiasm
        - Agreeableness: cooperation, trust, empathy
        - Neuroticism: emotional stability, stress response, anxiety levels

        • Use "they" as the pronoun. Do not mention the famous person's name
        • Focus on behavioral manifestations of each trait
        • Be specific about how this personality type would express each dimension
        • Avoid biographical details or specific achievements
        • Keep descriptions concrete and behavioral

        Format as valid JSON:
        {{
            "Openness": "description here",
            "Conscientiousness": "description here", 
            "Extraversion": "description here",
            "Agreeableness": "description here",
            "Neuroticism": "description here"
        }}"""
        
        response = self._make_api_call(prompt, max_tokens=400, temperature=0.7, model=self.model)
        
        if response:
            try:
                # Clean response - remove markdown code blocks if present
                cleaned_response = response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]  # Remove ```json
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]  # Remove ```
                cleaned_response = cleaned_response.strip()
                
                # Parse JSON response
                profiles = json.loads(cleaned_response)
                self.logger.info(f"Generated Big 5 profiles based on {famous_person_name}")
                return profiles
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON response for {famous_person_name}: {e}")
                self.logger.error(f"Raw response: {response[:200]}...")
                return None
        else:
            self.logger.error(f"Failed to generate Big 5 profiles for {famous_person_name}")
            return None

    def create_user_from_famous_person(self, famous_person_name: str, user_name: str = None) -> Optional[User]:
        """
        Create a complete User object based on a famous person's personality.
        
        Args:
            famous_person_name: Name of the famous person to base the profile on
            user_name: Name for the created user (defaults to "User inspired by [famous_person]")
            
        Returns:
            User object with generated profiles, or None if failed
        """
        # Generate user name if not provided
        if user_name is None:
            user_name = f"User inspired by {famous_person_name}"
        
        # Create user
        user = User(user_name)
        
        # Generate interests profile
        interests_profile = self.generate_profile_from_famous_person(famous_person_name)
        if interests_profile:
            user.set_interests_profile(interests_profile)
        else:
            self.logger.error(f"Failed to generate interests profile for {famous_person_name}")
            return None
        
        # Generate personality profiles
        personality_profiles = self.generate_personality_profiles_from_famous_person(famous_person_name)
        if personality_profiles:
            user.set_personality_profiles(personality_profiles)
        else:
            self.logger.error(f"Failed to generate personality profiles for {famous_person_name}")
            return None
        
        # Add metadata
        user.metadata = {
            'source': 'famous_person',
            'inspiration': famous_person_name,
            'generation_method': 'AI_profile_from_celebrity'
        }
        
        self.logger.info(f"Created user '{user_name}' based on {famous_person_name}")
        return user

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            "model": self.model,
            "total_requests": self.request_count,
            "archetypes_count": len(self.archetypes)
        }