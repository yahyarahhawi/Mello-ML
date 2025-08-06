import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_book_taste(books):
    """Generate personality description based on book preferences"""
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    # Format books for prompt
    book_list = ""
    for book in books:
        book_list += f"• {book['title']} by {book['author']} - {book['rating']} stars\n"
    
    prompt = f"""Based on the following list of books and ratings, write a 100-word paragraph that describes what this person's personality and inner world might be like.

{book_list}

• Do not mention any book titles, authors.
• You may mention high level topics. like (politics, palestine, psychology)
• Use "they" as the pronoun.
• Focus on what their reading preferences suggest about their character — such as how they think, what they value, and how they see the world.
• Make the paragraph semantically and emotionally rich, but avoid overly poetic language or complex vocabulary.
• Do not do literary analysis. This is a personality sketch inferred from taste.
• Think of it like reading between the lines of someone's bookshelf to understand who they are."""

    return make_api_call(prompt, api_key)

def generate_movie_taste(movies):
    """Generate personality description based on movie preferences"""
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    # Format movies for prompt
    movie_list = ""
    for movie in movies:
        movie_list += f"• {movie['title']} ({movie['year']}) - {movie['rating']} stars\n"
    
    prompt = f"""Based on the following list of movies and ratings, write a 100-word paragraph that describes what this person's personality and inner world might be like.

{movie_list}

• Do not mention any movie titles, directors, or actors.
• You may mention high level themes, genres, or topics.
• Use "they" as the pronoun.
• Focus on what their viewing preferences suggest about their character — such as how they think, what they value, and how they see the world.
• Make the paragraph semantically and emotionally rich, but avoid overly poetic language or complex vocabulary.
• Do not do film analysis. This is a personality sketch inferred from taste.
• Think of it like reading between the lines of someone's movie collection to understand who they are."""

    return make_api_call(prompt, api_key)

def generate_music_taste(songs):
    """Generate personality description based on music preferences"""
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    # Format songs for prompt
    song_list = ""
    for song in songs:
        song_list += f"• {song['title']} by {song['artist']}\n"
    
    prompt = f"""Based on the following list of songs, write a 100-word paragraph that describes what this person's personality and inner world might be like.

{song_list}

• Do not mention any song titles, artists, or albums.
• You may mention high level musical genres, themes, or moods.
• Use "they" as the pronoun.
• Focus on what their musical preferences suggest about their character — such as how they think, what they value, and how they see the world.
• Make the paragraph semantically and emotionally rich, but avoid overly poetic language or complex vocabulary.
• Do not do musical analysis. This is a personality sketch inferred from taste.
• Think of it like reading between the lines of someone's playlist to understand who they are."""

    return make_api_call(prompt, api_key)

def make_api_call(prompt, api_key):
    """Make API call to Gemini 2.5 Flash"""
    
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
        "temperature": 0.7,
        "max_tokens": 150
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
        return content
        
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def main():
    """Generate taste profiles for all users"""
    
    # Load existing users
    try:
        with open('raw_users.json', 'r', encoding='utf-8') as f:
            users = json.load(f)
    except FileNotFoundError:
        print("Error: raw_users.json not found. Run generate_raw_user.py first.")
        return
    except Exception as e:
        print(f"Error loading users: {e}")
        return
    
    taste_profiles = []
    
    print(f"Generating taste profiles for {len(users)} users...")
    
    for i, user in enumerate(users, 1):
        print(f"Processing user {i}/{len(users)}: {user.get('name', 'Unknown')}")
        
        # Generate taste descriptions
        book_taste = generate_book_taste(user.get('books', []))
        movie_taste = generate_movie_taste(user.get('movies', []))
        music_taste = generate_music_taste(user.get('songs', []))
        
        if book_taste and movie_taste and music_taste:
            profile = {
                "name": user.get('name'),
                "book_taste": book_taste,
                "movie_taste": movie_taste,
                "music_taste": music_taste
            }
            taste_profiles.append(profile)
            print(f"✓ Generated profile for {user.get('name')}")
        else:
            print(f"✗ Failed to generate profile for {user.get('name')}")
    
    # Save taste profiles
    output_file = "taste_profiles.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(taste_profiles, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Successfully saved {len(taste_profiles)} taste profiles to {output_file}")
    except Exception as e:
        print(f"✗ Failed to save taste profiles: {e}")

if __name__ == "__main__":
    main()