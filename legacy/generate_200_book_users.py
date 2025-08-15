#!/usr/bin/env python3
"""
Generate 200 diverse users with book preferences only.
Each user will have 10-15 books with ratings and a detailed personality profile.
"""

import os
import json
import random
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 50 diverse personality archetypes for variety
PERSONALITY_ARCHETYPES = [
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

def generate_user_profile(archetype, user_number, existing_users=None):
    """Generate a single user with book preferences based on personality archetype."""
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    # Extract commonly used books from existing users to moderate repetition
    common_guidance = ""
    if existing_users and len(existing_users) >= 5:
        book_counts = {}
        for user in existing_users:
            for book in user.get('books', []):
                title = book['title']
                book_counts[title] = book_counts.get(title, 0) + 1
        
        # Find books that appear in more than 60% of existing users (overly common)
        threshold = len(existing_users) * 0.6
        overused_books = [title for title, count in book_counts.items() if count >= threshold]
        
        if overused_books:
            common_guidance = f"\n\nDIVERSITY NOTE: These books are appearing too frequently: {', '.join(overused_books[:5])}. Include at most 1-2 of these if they truly fit the personality, but focus on other books that match the archetype."
    
    prompt = f"""Generate a realistic college student profile with coherent taste in books only. Return ONLY valid JSON in this exact format:

{{
  "name": "First Last",
  "books": [
    {{"title": "Book Title", "author": "Author Name", "rating": 4}},
    // 20 books total with ratings 1-5, include a mix of ratings (not all 4-5 stars)
  ]
}}

Create a student who is a {archetype}. Their books should authentically reflect this personality type. Include a mix of:
- Both fiction AND non-fiction books (philosophy, self-help, religion, biography, science, history, etc.)
- Some popular/well-known works that fit the archetype (for realistic overlap with other users)
- Some lesser-known gems that show unique taste
- A realistic rating distribution (include 2-3 star ratings, not everything highly rated)

The goal is authentic personality-driven taste across all genres, not just literary fiction.{common_guidance}

Generate user #{user_number}:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "google/gemini-2.5-flash",
        "messages": [
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.8,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        
        # Clean up the response to ensure it's valid JSON
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        
        user_data = json.loads(content)
        return user_data
        
    except Exception as e:
        print(f"Error generating user {user_number}: {e}")
        return None

def format_user_data(user_data, archetype):
    """Format user data to match expected structure."""
    
    # Ensure we have at least some books
    if not user_data.get('books') or len(user_data['books']) < 5:
        default_books = [
            {"title": "The Alchemist", "author": "Paulo Coelho", "rating": 4},
            {"title": "Atomic Habits", "author": "James Clear", "rating": 5},
            {"title": "Educated", "author": "Tara Westover", "rating": 4},
            {"title": "The Seven Husbands of Evelyn Hugo", "author": "Taylor Jenkins Reid", "rating": 4},
            {"title": "Sapiens", "author": "Yuval Noah Harari", "rating": 5}
        ]
        if not user_data.get('books'):
            user_data['books'] = []
        user_data['books'].extend(default_books[:20-len(user_data['books'])])
    
    return {
        "name": user_data.get('name', f"Reader {random.randint(1000, 9999)}"),
        "personality_archetype": archetype,
        "books": user_data['books'][:20]  # Limit to 20 books as per prompt
    }

def main():
    print("🔥 Generating 200 diverse book readers...")
    
    users = []
    
    # First, add Yahya Rahhawi from raw_real_users.json
    try:
        with open('raw_real_users.json', 'r') as f:
            real_users = json.load(f)
            yahya = real_users[0]  # Yahya is the first user
            yahya_books = {
                "name": yahya["name"],
                "personality_archetype": "Contemplative reader interested in philosophy, existential questions, and social justice",
                "books": yahya["books"]
            }
            users.append(yahya_books)
            print(f"  ✓ Added Yahya Rahhawi ({len(yahya_books['books'])} books)")
    except (FileNotFoundError, IndexError) as e:
        print(f"  ⚠️ Could not load Yahya's profile: {e}")
    
    # Generate 199 more users (4 users per archetype for remaining users)
    for i in range(199):
        archetype = PERSONALITY_ARCHETYPES[i % len(PERSONALITY_ARCHETYPES)]
        
        print(f"Generating user {i+2}/200: {archetype[:50]}...")
        
        user_data = generate_user_profile(archetype, i+2, users)
        if user_data:
            formatted_user = format_user_data(user_data, archetype)
            users.append(formatted_user)
            print(f"  ✓ Created: {formatted_user['name']} ({len(formatted_user['books'])} books)")
        else:
            print(f"  ✗ Failed to generate user {i+2}")
    
    # Save to file
    output_file = "book_users_200.json"
    with open(output_file, 'w') as f:
        json.dump(users, f, indent=2)
    
    print(f"\n🎉 Successfully generated {len(users)} users!")
    print(f"📁 Saved to: {output_file}")
    print(f"📊 Total books in dataset: {sum(len(user['books']) for user in users)}")

if __name__ == "__main__":
    main()