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
        
        # Comprehensive personality archetypes for diverse user generation (50 types)
        # Each archetype defines a complete personality profile, not just reading preferences
        self.personality_archetypes = [
    "High Openness, High Conscientiousness academic: loves learning, organized, seeks novel ideas but executes systematically. Values intellectual growth and structured exploration.",
    "High Openness, Low Conscientiousness creative: artistic, spontaneous, generates ideas constantly but struggles with follow-through. Thrives on inspiration and freedom.",
    "Low Openness, High Conscientiousness traditionalist: prefers familiar routines, highly reliable, values stability and proven methods. Excels at maintaining systems.",
    "High Extraversion, High Agreeableness social connector: energized by people, naturally collaborative, builds bridges between groups. Seeks harmony and shared experiences.",
    "High Extraversion, Low Agreeableness ambitious leader: confident, competitive, drives results, comfortable with conflict. Focuses on achievement and influence.",
    "Low Extraversion, High Agreeableness supportive listener: prefers small groups, deeply empathetic, provides quiet support. Values authentic one-on-one connections.",
    "Low Extraversion, High Openness introspective explorer: enjoys solitude for deep thinking, curious about ideas, processes internally. Seeks meaning and understanding.",
    "High Neuroticism, High Conscientiousness perfectionist: anxious about details, works hard to control outcomes, seeks security through preparation. Worries but delivers.",
    "Low Neuroticism, High Extraversion confident optimist: naturally resilient, socially bold, bounces back quickly from setbacks. Maintains positive energy.",
    "High Openness, Low Agreeableness independent thinker: questions conventions, comfortable disagreeing, values authenticity over harmony. Thinks for themselves.",
    "Low Openness, High Agreeableness loyal traditionalist: values family and community traditions, supportive of established ways, prioritizes group cohesion over change.",
    "High Conscientiousness, Low Neuroticism steady achiever: calm under pressure, follows through reliably, maintains consistent performance. Natural project manager.",
    "Low Conscientiousness, High Extraversion spontaneous socializer: lives in the moment, energized by people and new experiences, adaptable but sometimes unreliable.",
    "High Neuroticism, Low Conscientiousness anxious procrastinator: worries about tasks but struggles to start them, seeks comfort and validation, overwhelmed by details.",
    "High Agreeableness, High Conscientiousness dedicated helper: puts others first, follows through on commitments to help, naturally service-oriented and reliable.",
    "Low Agreeableness, Low Neuroticism tough pragmatist: emotionally resilient, direct communicator, focused on practical results over relationships.",
    "High Openness, High Extraversion enthusiastic innovator: loves sharing new ideas with others, energized by creative collaboration, seeks novel social experiences.",
    "Low Extraversion, Low Agreeableness analytical loner: prefers working alone, skeptical of others' motives, values competence over social connection.",
    "High Conscientiousness, High Agreeableness responsible caregiver: dutiful, puts family/community needs first, maintains harmony through consistent support.",
    "Low Conscientiousness, Low Agreeableness rebellious individualist: resists rules and expectations, prioritizes personal freedom, questions authority naturally.",
    "High Neuroticism, High Agreeableness sensitive empath: deeply affected by others' emotions, worries about relationships, seeks to please and avoid conflict.",
    "Low Neuroticism, Low Extraversion calm introvert: emotionally stable, prefers solitude, processes quietly, maintains equilibrium in chaos.",
    "High Extraversion, High Conscientiousness organized leader: energized by managing teams, follows systems while motivating others, natural organizer of group activities.",
    "Low Extraversion, High Conscientiousness methodical specialist: prefers working alone on detailed projects, thorough and systematic, values expertise and precision.",
    "High Openness, High Neuroticism creative worrier: generates many ideas but anxious about execution, sensitive to criticism, seeks artistic expression as outlet.",
    "Low Openness, Low Neuroticism practical stabilizer: prefers tried-and-true methods, emotionally steady, provides calm consistency in uncertain situations.",
    "High Agreeableness, Low Conscientiousness warm but scattered: genuinely cares about people, wants to help everyone, but struggles with time management and priorities.",
    "Low Agreeableness, High Conscientiousness demanding perfectionist: high standards for self and others, focused on results, comfortable giving tough feedback.",
    "High Extraversion, Low Conscientiousness charismatic improviser: naturally draws people in, great at adapting in the moment, but inconsistent with commitments.",
    "Low Extraversion, Low Conscientiousness quiet dreamer: introspective, imaginative, goes with the flow, values freedom over achievement or social connection.",
    "High Neuroticism, Low Extraversion worried observer: anxious in social situations, prefers familiar environments, overthinks interactions, seeks security.",
    "Low Neuroticism, High Openness adventurous explorer: emotionally resilient, seeks new experiences, comfortable with uncertainty, naturally optimistic about change.",
    "High Conscientiousness, Low Openness disciplined traditionalist: maintains established routines efficiently, values proven methods, excels at consistent execution.",
    "Low Conscientiousness, High Openness creative free spirit: generates novel ideas constantly, explores many interests, struggles with conventional structure.",
    "High Agreeableness, High Extraversion popular harmonizer: naturally likeable, energized by social connection, works to maintain group morale and inclusion.",
    "Low Agreeableness, Low Extraversion skeptical individualist: questions others' motives, prefers independence, values competence and self-reliance over social bonds.",
    "High Neuroticism, High Extraversion dramatic performer: emotionally expressive, seeks attention and validation, energized by social reactions, mood varies with audience.",
    "Low Neuroticism, Low Agreeableness confident competitor: emotionally stable, comfortable with conflict, focused on winning, doesn't worry about others' opinions.",
    "High Openness, Low Extraversion thoughtful philosopher: enjoys deep contemplation alone, curious about abstract ideas, processes internally before sharing insights.",
    "Low Openness, High Extraversion social traditionalist: energized by familiar people and activities, maintains established social connections, prefers known routines.",
    "High Conscientiousness, High Neuroticism anxious overachiever: worries about meeting high standards, works harder when stressed, seeks control through preparation.",
    "Low Conscientiousness, Low Neuroticism relaxed underachiever: doesn't stress about deadlines or standards, goes with the flow, prioritizes immediate comfort.",
    "High Agreeableness, Low Extraversion gentle supporter: cares deeply but expresses it quietly, prefers helping behind the scenes, avoids spotlight.",
    "Low Agreeableness, High Extraversion bold challenger: confident in disagreeing publicly, comfortable with confrontation, pushes others to perform better.",
    "High Neuroticism, Low Agreeableness defensive worrier: anxious about others' motives, quick to feel criticized, protects self through emotional distance.",
    "Low Neuroticism, High Agreeableness easygoing collaborator: naturally trusting, doesn't take things personally, maintains positive relationships effortlessly.",
    "High Openness, High Agreeableness idealistic connector: curious about people's perspectives, seeks to understand and bridge differences, values diversity.",
    "Low Openness, Low Agreeableness practical skeptic: prefers established ways, questions new ideas, focused on practical results over relationships or innovation.",
    "Balanced moderate: moderate on all traits, adapts personality to situation, can relate to many different types of people, seeks middle ground.",
    "High variability complex: shows different trait combinations in different contexts, unpredictable personality patterns, influenced heavily by environment and mood."
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
    
    def _generate_profile_section(self, name: str, personality_type: str) -> Optional[dict]:
        """Generate profile section with basic info."""
        prompt = f"""For a college student with this personality: "{personality_type}"

Generate ONLY their basic profile info. Choose a realistic major and write a bio in their authentic voice:

Name: {name}
Major: [choose realistic major that fits personality]
Class Year: [2024, 2025, 2026, or 2027] 
Bio: [2-3 sentences this person would write about themselves - authentic voice]
Interests: [list 5-7 genuine interests this personality would have]

Format as: Major|ClassYear|Bio|Interest1,Interest2,Interest3...

Example: Computer Science|2025|Love building apps and exploring AI. Always learning something new!|Programming,AI,Gaming,Sci-fi,Music"""
        
        response = self._make_api_call(prompt, self.user_generation_model, max_tokens=200, temperature=1.2)
        
        if not response:
            return None
            
        try:
            parts = response.strip().split('|')
            if len(parts) >= 4:
                major = parts[0].strip()
                class_year = parts[1].strip()
                bio = parts[2].strip()
                interests = [i.strip() for i in parts[3].split(',')]
                
                return {
                    "fullName": name,
                    "classYear": class_year,
                    "major": major,
                    "bio": bio,
                    "interests": interests
                }
        except Exception as e:
            self.logger.error(f"Error parsing profile section: {e}")
            
        return None
    
    def _generate_books_section(self, personality_type: str) -> Optional[dict]:
        """Generate books section."""
        prompt = f"""For someone with this personality: "{personality_type}"

List 5 books they would genuinely love. Use REAL book titles and authors.

Format each as: Title by Author
Then write 3 brief reviews and a reflection.

Books:
1. [Book Title] by [Author]
2. [Book Title] by [Author]
3. [Book Title] by [Author]  
4. [Book Title] by [Author]
5. [Book Title] by [Author]

Review for book 1: [1-2 sentences in their voice]
Review for book 2: [1-2 sentences in their voice]
Review for book 3: [1-2 sentences in their voice]

Reflection: [Why they choose these books - 1-2 sentences]"""
        
        response = self._make_api_call(prompt, self.user_generation_model, max_tokens=400, temperature=1.2)
        
        if not response:
            return None
            
        try:
            lines = response.strip().split('\n')
            books = []
            reviews = {}
            reflection = ""
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    # Extract book title
                    book_line = line.split('.', 1)[1].strip()
                    if ' by ' in book_line:
                        book_title = book_line.split(' by ')[0].strip()
                        books.append(book_title)
                        
                elif line.startswith('Review for book'):
                    # Extract review - handle various formats
                    if ':' in line:
                        review_text = line.split(':', 1)[1].strip()
                        # Extract book number more robustly
                        try:
                            # Try to find number in the line
                            parts = line.split()
                            book_num = None
                            for part in parts:
                                if part.replace(':', '').isdigit():
                                    book_num = int(part.replace(':', '')) - 1
                                    break
                            
                            if book_num is not None and 0 <= book_num < len(books):
                                reviews[books[book_num]] = review_text
                        except (ValueError, IndexError):
                            # If parsing fails, just use the first book as fallback
                            if books:
                                reviews[books[0]] = review_text
                            
                elif line.startswith('Reflection:'):
                    reflection = line.split(':', 1)[1].strip()
            
            if len(books) >= 3:
                return {
                    "favoriteBooks": books,
                    "bookReviews": reviews,
                    "bookReflection": reflection
                }
                
        except Exception as e:
            self.logger.error(f"Error parsing books section: {e}")
            
        return None
    
    def _generate_movies_section(self, personality_type: str) -> Optional[dict]:
        """Generate movies section."""
        prompt = f"""For someone with this personality: "{personality_type}"

List 5 movies/shows they would genuinely love. Use REAL titles.

Format:
Movies:
1. [Movie/Show Title]
2. [Movie/Show Title]  
3. [Movie/Show Title]
4. [Movie/Show Title]
5. [Movie/Show Title]

Review for movie 1: [1-2 sentences in their voice]
Review for movie 2: [1-2 sentences in their voice]
Review for movie 3: [1-2 sentences in their voice]

Reflection: [Why they're drawn to these films - 1-2 sentences]"""
        
        response = self._make_api_call(prompt, self.user_generation_model, max_tokens=400, temperature=1.2)
        
        if not response:
            return None
            
        try:
            lines = response.strip().split('\n')
            movies = []
            reviews = {}
            reflection = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    movie_title = line.split('.', 1)[1].strip()
                    movies.append(movie_title)
                    
                elif line.startswith('Review for movie'):
                    if ':' in line:
                        review_text = line.split(':', 1)[1].strip()
                        # Extract movie number more robustly
                        try:
                            # Try to find number in the line
                            parts = line.split()
                            movie_num = None
                            for part in parts:
                                if part.replace(':', '').isdigit():
                                    movie_num = int(part.replace(':', '')) - 1
                                    break
                            
                            if movie_num is not None and 0 <= movie_num < len(movies):
                                reviews[movies[movie_num]] = review_text
                        except (ValueError, IndexError):
                            # If parsing fails, just use the first movie as fallback
                            if movies:
                                reviews[movies[0]] = review_text
                            
                elif line.startswith('Reflection:'):
                    reflection = line.split(':', 1)[1].strip()
            
            if len(movies) >= 3:
                return {
                    "favoriteMovies": movies,
                    "movieReviews": reviews,
                    "movieReflection": reflection
                }
                
        except Exception as e:
            self.logger.error(f"Error parsing movies section: {e}")
            
        return None
    
    def _generate_music_section(self, personality_type: str) -> Optional[dict]:
        """Generate music section."""
        prompt = f"""For someone with this personality: "{personality_type}"

List 5 music artists/bands they would genuinely love. Use REAL artist names.
Then describe their music taste in their authentic voice.

Artists:
1. [Artist/Band]
2. [Artist/Band]  
3. [Artist/Band]
4. [Artist/Band]
5. [Artist/Band]

Vibe: [How they'd describe their music taste - 1-2 sentences in their voice]"""
        
        response = self._make_api_call(prompt, self.user_generation_model, max_tokens=200, temperature=1.2)
        
        if not response:
            return None
            
        try:
            lines = response.strip().split('\n')
            artists = []
            vibe = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    artist = line.split('.', 1)[1].strip()
                    artists.append(artist)
                    
                elif line.startswith('Vibe:'):
                    vibe = line.split(':', 1)[1].strip()
            
            if len(artists) >= 3:
                return {
                    "musicArtists": artists,
                    "vibeMatch": vibe
                }
                
        except Exception as e:
            self.logger.error(f"Error parsing music section: {e}")
            
        return None
    
    def _generate_personality_section(self, personality_type: str, user_context: str) -> Optional[dict]:
        """Generate Big 5 personality responses ensuring proper selection distribution."""
        # Complete options for each trait (from the React frontend)
        trait_options = {
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
        
        personality_data = {}
        
        # Generate responses for each trait with context
        for trait in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']:
            trait_level = "High" if trait in personality_type else "Moderate"
            if f"Low {trait}" in personality_type:
                trait_level = "Low"
            
            prompt = f"""CONTEXT: {user_context}
PERSONALITY: "{personality_type}"

For someone with {trait_level} {trait}, select 4-6 behaviors that authentically describe them:

{chr(10).join([f"- {option}" for option in trait_options.get(trait, [])])}

REQUIREMENTS:
- Must select exactly 4-6 behaviors that fit their {trait_level} {trait} level
- Consider their background: {user_context}
- Match their personality archetype: {personality_type}

Format your response exactly like this:
Selected:
- [behavior 1 exactly as written above]
- [behavior 2 exactly as written above]
- [behavior 3 exactly as written above]
- [behavior 4 exactly as written above]
- [behavior 5 exactly as written above]
- [behavior 6 exactly as written above]

Note: [Brief note in their voice about their {trait}]"""
            
            response = self._make_api_call(prompt, self.user_generation_model, max_tokens=300, temperature=1.0)
            
            if response:
                try:
                    lines = response.strip().split('\n')
                    selected = []
                    note = ""
                    in_selected = False
                    
                    for line in lines:
                        line = line.strip()
                        if line.startswith('Selected:'):
                            in_selected = True
                            continue
                        elif line.startswith('Note:'):
                            note = line.split(':', 1)[1].strip()
                            in_selected = False
                        elif in_selected and line.startswith('-'):
                            behavior = line[1:].strip()
                            selected.append(behavior)
                    
                    # Get all options for this trait
                    all_options = trait_options.get(trait, [])
                    not_selected = [opt for opt in all_options if opt not in selected]
                    
                    personality_data[trait] = {
                        "selected": selected,
                        "not_selected": not_selected,
                        "notes": note
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error parsing {trait} section: {e}")
        
        return personality_data if personality_data else None
    
    def _generate_additional_info_section(self, personality_type: str) -> Optional[dict]:
        """Generate additional info section."""
        prompt = f"""For someone with this personality: "{personality_type}"

Answer these questions in their authentic voice:

1. What could you talk about for hours?
2. What flows naturally in your conversations?
3. What do you reach for in your alone time?
4. Describe your perfect day
5. What energizes you? (list 3-4 things)

Format as:
Talk about: [answer 1]
Conversations: [answer 2] 
Alone time: [answer 3]
Perfect day: [answer 4]
Energy sources: [item1, item2, item3, item4]"""
        
        response = self._make_api_call(prompt, self.user_generation_model, max_tokens=300, temperature=1.2)
        
        if not response:
            return None
            
        try:
            lines = response.strip().split('\n')
            additional_info = {}
            
            for line in lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'Talk about':
                        additional_info['talkAboutForHours'] = value
                    elif key == 'Conversations':
                        additional_info['naturalConversations'] = value
                    elif key == 'Alone time':
                        additional_info['aloneTimeReachFor'] = value
                    elif key == 'Perfect day':
                        additional_info['perfectDay'] = value
                    elif key == 'Energy sources':
                        additional_info['energySources'] = [item.strip() for item in value.split(',')]
            
            return additional_info
            
        except Exception as e:
            self.logger.error(f"Error parsing additional info section: {e}")
            
        return None
    
    def generate_synthetic_user_json(self, personality_type: str = None) -> Optional[dict]:
        """
        Generate realistic frontend JSON data by building it manually through multiple LLM calls.
        This prevents JSON parsing errors and ensures robust generation.
        Each call uses context from previous calls for coherence.
        
        Args:
            personality_type: Specific personality archetype, or random if None
            
        Returns:
            Dictionary in frontend JSON format (or None if generation failed)
        """
        if personality_type is None:
            personality_type = random.choice(self.personality_archetypes)
        
        # Generate realistic name using Faker
        if self.fake:
            name = self.fake.name()
        else:
            name = f"User_{random.randint(1000, 9999)}"
        
        self.logger.info(f"Building JSON manually for {name} ({personality_type[:30]}...)")
        
        # Step 1: Generate profile section (establishes core identity)
        profile_section = self._generate_profile_section(name, personality_type)
        if not profile_section:
            self.logger.error(f"Failed to generate profile section for {name}")
            return None
        
        # Build user context for subsequent calls
        user_context = f"Name: {name}, Major: {profile_section['major']}, Bio: {profile_section['bio']}, Interests: {', '.join(profile_section['interests'])}"
        
        # Step 2-6: Generate other sections with context for coherence
        books_section = self._generate_books_section(personality_type) 
        movies_section = self._generate_movies_section(personality_type)
        music_section = self._generate_music_section(personality_type)
        personality_section = self._generate_personality_section(personality_type, user_context)
        additional_info_section = self._generate_additional_info_section(personality_type)
        
        # Check if all sections were generated successfully
        missing_sections = []
        if not books_section: missing_sections.append("books")
        if not movies_section: missing_sections.append("movies") 
        if not music_section: missing_sections.append("music")
        if not personality_section: missing_sections.append("personality")
        if not additional_info_section: missing_sections.append("additionalInfo")
        
        if missing_sections:
            self.logger.error(f"Failed to generate sections for {name}: {missing_sections}")
            return None
        
        # Assemble complete JSON manually (no parsing errors!)
        user_json = {
            "profile": profile_section,
            "books": books_section,
            "movies": movies_section,
            "music": music_section,
            "personality": personality_section,
            "additionalInfo": additional_info_section,
            "metadata": {
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0"
            }
        }
        
        self.logger.info(f"Successfully built complete JSON for {name}")
        return user_json
    
    def generate_synthetic_user(self, personality_type: str = None) -> Optional[User]:
        """
        Generate a synthetic user by creating realistic JSON then processing through
        the same pipeline that real users go through.
        
        Args:
            personality_type: Specific personality archetype, or random if None
            
        Returns:
            User object processed through standard pipeline
        """
        # Step 1: Generate realistic JSON data
        user_json = self.generate_synthetic_user_json(personality_type)
        
        if not user_json:
            return None
        
        try:
            # Step 2: Create user from JSON (same as real users)
            user = User.from_frontend_json(user_json)
            
            # Add metadata about generation
            user.metadata['personality_type'] = personality_type
            user.metadata['generated'] = True
            user.metadata['generation_method'] = 'realistic_json_pipeline'
            
            name = user.name
            self.logger.info(f"Created synthetic user from JSON: {name}")
            
            # NOTE: Profile generation and embeddings happen separately
            # This user will go through the same dual-vector pipeline as real users:
            # 1. generate_dual_profiles() -> interests_profile + personality_profile  
            # 2. embed_user_dual() -> interests_embedding + personality_embedding
            
            return user
            
        except Exception as e:
            self.logger.error(f"Error creating user from generated JSON: {e}")
            return None
    
    def generate_taste_profile(self, user: User) -> bool:
        """
        LEGACY METHOD - Generate an AI taste profile for a user based on their book preferences.
        
        NOTE: This method is deprecated. Use generate_dual_profiles() instead.
        
        Args:
            user: User to generate profile for
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.warning(f"Using legacy generate_taste_profile for {user.name}. Consider using generate_dual_profiles() instead.")
        
        if not user.preferences or not user.preferences.books:
            self.logger.error(f"User {user.name} has no books to analyze")
            return False
        
        # Create book summary
        book_summary = f"{user.name}'s book preferences:\n"
        for book in user.preferences.books:
            book_summary += f"- {book.title} by {book.author}: {book.rating}/5 stars ({book.genre})\n"
        
        # Create prompt for taste profile generation
        prompt = f""" {book_summary}
        Based on the following list of books, write a 200-word personality profile that captures how this person thinks, feels, and behaves.
        • Use "they" as the pronoun. do not mention name
        • Focus on inferring personality traits from the books, including cognitive style, emotional tendencies, values, motivations, and social orientation.
        • Use clear, semantically rich language with personality-relevant descriptors (e.g., analytical, empathetic, adventurous, introverted, pragmatic, idealistic, resilient).
        • Avoid listing titles, authors, or specific book names.
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
            self.logger.info(f"Generated legacy taste profile for {user.name}")
            return True
        else:
            self.logger.error(f"Failed to generate legacy taste profile for {user.name}")
            return False
    
    def generate_interests_profile(self, user: User) -> bool:
        """
        Generate an AI interests profile based on media preferences and profile data.
        
        Args:
            user: User to generate profile for
            
        Returns:
            True if successful, False otherwise
        """
        # Build comprehensive interests summary
        profile_data = user.profile_data
        # Get media data from the original frontend JSON structure stored in metadata
        frontend_data = user.metadata.get('frontend_data', {})
        books_data = frontend_data.get('books', {})
        movies_data = frontend_data.get('movies', {})
        music_data = frontend_data.get('music', {})
        additional_info = user.additional_info
        
        # Create input for interests profile
        input_text = f"""Profile Information:
- Name: {profile_data.get('fullName', user.name)}
- Major: {profile_data.get('major', '')}
- Bio: {profile_data.get('bio', '')}
- Interests: {', '.join(profile_data.get('interests', []))}

Favorite Books: {', '.join(books_data.get('favoriteBooks', []))}

Book Reviews:
"""
        for book, review in books_data.get('bookReviews', {}).items():
            input_text += f"- {book}: {review[:200]}...\n"
        
        input_text += f"\nBook Reflection: {books_data.get('bookReflection', '')}\n"
        
        input_text += f"\nFavorite Movies: {', '.join(movies_data.get('favoriteMovies', []))}\n"
        
        input_text += "\nMovie Reviews:\n"
        for movie, review in movies_data.get('movieReviews', {}).items():
            input_text += f"- {movie}: {review[:200]}...\n"
        
        input_text += f"\nMovie Reflection: {movies_data.get('movieReflection', '')}\n"
        
        input_text += f"\nMusic Artists: {', '.join(music_data.get('musicArtists', []))}\n"
        input_text += f"Music Vibe: {music_data.get('vibeMatch', '')}\n"
        
        input_text += f"\nWhat they could talk about for hours: {additional_info.get('talkAboutForHours', '')}\n"
        input_text += f"Natural conversations: {additional_info.get('naturalConversations', '')}\n"
        input_text += f"Alone time activities: {additional_info.get('aloneTimeReachFor', '')}\n"
        input_text += f"Perfect day: {additional_info.get('perfectDay', '')}\n"
        
        prompt = f"""Input:
{input_text}

Write a concise 200 word interests summary for a person based on their favorite books, films, music, and additional personal notes.
• Use these works and notes as evidence to identify major themes in their cultural and intellectual tastes.
• Avoid listing every title or artist; instead, describe the patterns these choices reveal.
• Use concrete, personality-relevant descriptors (e.g., intellectually curious, socially aware, visually oriented).
• Do not speculate about unrelated traits such as work habits unless strongly implied.
• Make description meaningfully rich and semantically rich. Do not repeat yourself unless to emphasize a significant trait.
• Use "they" pronouns and avoid mentioning the person's name.
• Output exactly one paragraph in plain text, without headings or formatting."""
        
        response = self._make_api_call(prompt, self.taste_profile_model, max_tokens=250, temperature=1.2)
        
        if response:
            user.set_interests_profile(response)
            self.logger.info(f"Generated interests profile for {user.name}")
            return True
        else:
            self.logger.error(f"Failed to generate interests profile for {user.name}")
            return False
    
    def generate_personality_profiles(self, user: User) -> bool:
        """
        Generate AI personality profiles for each Big 5 trait based on selected/not-selected behaviors.
        
        Args:
            user: User to generate profiles for
            
        Returns:
            True if successful, False otherwise
        """
        personality_responses = user.personality_responses
        if not personality_responses:
            self.logger.error(f"User {user.name} has no personality responses")
            return False
        
        trait_profiles = {}
        traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        
        for trait in traits:
            if trait not in personality_responses:
                continue
                
            trait_data = personality_responses[trait]
            selected = trait_data.get('selected', [])
            not_selected = trait_data.get('not_selected', [])
            notes = trait_data.get('notes', '')
            
            # Create behavior examples for prompt
            examples_text = "Selected behaviors (what they do):\n"
            for behavior in selected:
                examples_text += f"• {behavior}\n"
            
            examples_text += "\nNot-selected behaviors (what they don't do):\n"
            for behavior in not_selected:
                examples_text += f"• {behavior}\n"
            
            if notes:
                examples_text += f"\nAdditional notes: {notes}\n"
            
            prompt = f"""Based only on the provided behavior examples, write a 150 word description of how this person expresses {trait} in daily life.

{examples_text}

• Treat selected examples as strong evidence of tendencies, and not-selected examples as weak evidence of the opposite.
• Use clear, behavior-focused language that makes their patterns obvious without naming the trait.
• Avoid moral judgments or value-laden praise.
• Keep it specific and concrete, reflecting real habits and preferences.
• Make it semantically rich and meaningful.
• Use "they" pronouns and focus on observable patterns.
• Output exactly one paragraph in plain text, without headings or formatting."""
            
            response = self._make_api_call(prompt, self.taste_profile_model, max_tokens=200, temperature=1.2)
            
            if response:
                trait_profiles[trait] = response
                self.logger.info(f"Generated {trait} profile for {user.name}")
            else:
                self.logger.error(f"Failed to generate {trait} profile for {user.name}")
                return False
        
        if trait_profiles:
            user.set_personality_profile(trait_profiles)
            self.logger.info(f"Generated all personality profiles for {user.name}")
            return True
        else:
            self.logger.error(f"Failed to generate any personality profiles for {user.name}")
            return False
    
    def generate_dual_profiles(self, user: User) -> bool:
        """
        Generate both interests and personality profiles for a user.
        
        Args:
            user: User to generate profiles for
            
        Returns:
            True if both profiles were generated successfully
        """
        interests_success = self.generate_interests_profile(user)
        personality_success = self.generate_personality_profiles(user)
        
        return interests_success and personality_success
    
    def generate_population(self, size: int = 50, progress_callback=None) -> Population:
        """
        Generate a population of realistic synthetic users from personality archetypes.
        These users are created with realistic JSON data (like real users would input)
        and then processed through the same pipeline.
        
        Args:
            size: Number of users to generate
            progress_callback: Function to call with progress updates
            
        Returns:
            Population with generated users (ready for dual-vector pipeline)
        """
        population = Population(f"Generated Population ({size} users)")
        
        self.logger.info(f"Generating {size} realistic synthetic users from personality archetypes...")
        self.logger.info("Step 1: Generate realistic JSON data (like real users would input)")
        self.logger.info("Step 2: Process through same pipeline as real users")
        
        # Ensure diverse personality types
        personality_cycle = (self.personality_archetypes * ((size // len(self.personality_archetypes)) + 1))[:size]
        random.shuffle(personality_cycle)
        
        successful_generations = 0
        
        for i in range(size):
            personality_type = personality_cycle[i]
            
            # Generate user from realistic JSON (same process as real users)
            user = self.generate_synthetic_user(personality_type)
            
            if user:
                population.add_user(user)
                successful_generations += 1
                archetype_short = personality_type.split(':')[0] if ':' in personality_type else personality_type[:30]
                self.logger.info(f"Generated user {successful_generations}: {user.name} ({archetype_short})")
            else:
                self.logger.warning(f"Failed to generate user {i + 1} with archetype: {personality_type[:50]}...")
            
            if progress_callback:
                progress_callback(i + 1, size)
            
            # Log progress every 3 users (JSON generation takes time)
            if (i + 1) % 3 == 0:
                self.logger.info(f"Progress: {successful_generations}/{i + 1} users generated")
        
        self.logger.info(f"✅ Successfully generated {successful_generations}/{size} realistic users")
        self.logger.info("Next steps: generate_dual_profiles_for_population() -> embed_population_dual()")
        return population
    
    def generate_dual_profiles_for_population(self, population: Population, progress_callback=None) -> int:
        """
        Generate dual profiles (interests + personality) for all users in a population.
        
        Args:
            population: Population to generate profiles for
            progress_callback: Function to call with progress updates
            
        Returns:
            Number of successfully generated dual profiles
        """
        users_without_dual_profiles = []
        for user in population.users:
            if not user.interests_profile or not user.personality_profile:
                users_without_dual_profiles.append(user)
        
        if not users_without_dual_profiles:
            self.logger.info("All users already have dual profiles")
            return 0
        
        self.logger.info(f"Generating dual profiles for {len(users_without_dual_profiles)} users")
        
        successful_profiles = 0
        
        for i, user in enumerate(users_without_dual_profiles):
            if self.generate_dual_profiles(user):
                successful_profiles += 1
                self.logger.info(f"Generated dual profiles for {user.name}")
            else:
                self.logger.warning(f"Failed to generate dual profiles for {user.name}")
            
            if progress_callback:
                progress_callback(i + 1, len(users_without_dual_profiles))
        
        self.logger.info(f"Successfully generated dual profiles for {successful_profiles}/{len(users_without_dual_profiles)} users")
        return successful_profiles
    
    def generate_taste_profiles_for_population(self, population: Population, progress_callback=None) -> int:
        """
        Generate legacy taste profiles for all users in a population.
        Note: This is for backward compatibility. New workflows should use dual profiles.
        
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
        
        self.logger.info(f"Generating legacy taste profiles for {len(users_without_profiles)} users")
        
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