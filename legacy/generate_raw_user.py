import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_user_data(user_number, existing_users=None):
    """Generate coherent user data with books, movies, and songs for one user"""
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    # Define personality archetypes to ensure diversity
    archetypes = [
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
        "art student passionate about visual storytelling and experimental media"
    ]
    
    archetype = archetypes[(user_number - 1) % len(archetypes)]
    
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
    
    prompt = f"""Generate a realistic college student profile with coherent taste across books, movies, and music. Return ONLY valid JSON in this exact format:

{{
  "name": "First Last",
  "books": [
    {{"title": "Book Title", "author": "Author Name", "rating": 4}},
    // 20 books total with ratings 1-5, include a mix of ratings (not all 4-5 stars)
  ],
  "movies": [
    {{"title": "Movie Title", "year": 2020, "rating": 4}},
    // 20 movies total with ratings 1-5, include a mix of ratings
  ],
  "songs": [
    {{"title": "Song Title", "artist": "Artist Name"}},
    // 20 songs total
  ]
}}

Create a student who is a {archetype}. Their books, movies, and music should authentically reflect this personality type. Include a mix of:
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
        "model": "moonshotai/kimi-k2",
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
        
    except requests.exceptions.RequestException as e:
        print(f"API request failed for user {user_number}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed for user {user_number}: {e}")
        print(f"Raw content: {content}")
        return None
    except Exception as e:
        print(f"Unexpected error for user {user_number}: {e}")
        return None

def main():
    """Generate 20 users and save to raw_users.json"""
    
    users = []
    
    print("Generating 20 users with diverse, coherent taste profiles...")
    
    for i in range(1, 21):
        print(f"Generating user {i}/20...")
        user_data = generate_user_data(i, users)  # Pass existing users for diversity
        
        if user_data:
            users.append(user_data)
            print(f"✓ Successfully generated user: {user_data.get('name', 'Unknown')}")
        else:
            print(f"✗ Failed to generate user {i}")
    
    # Save all users to JSON file
    output_file = "raw_users.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Successfully saved {len(users)} users to {output_file}")
    except Exception as e:
        print(f"✗ Failed to save users to file: {e}")

if __name__ == "__main__":
    main()