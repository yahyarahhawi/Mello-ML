#!/usr/bin/env python3
"""
Generate taste profiles for the 200 book users.
Creates 100-word personality descriptions based on their book preferences.
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def make_api_call(prompt, api_key):
    """Make API call using same pattern as generate_taste_profiles.py"""
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
        return result['choices'][0]['message']['content'].strip()
        
    except Exception as e:
        print(f"API call error: {e}")
        return None

def generate_taste_profile(user_data):
    """Generate a taste profile for a user based on their books."""
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    # Format books for prompt - exact same format as generate_taste_profiles.py
    book_list = ""
    for book in user_data['books']:
        title = book.get('title', 'Unknown Title')
        author = book.get('author', 'Unknown Author')
        rating = book.get('rating', 4)
        book_list += f"• {title} by {author} - {rating} stars\n"
    
    # Use exact same prompt as generate_taste_profiles.py
    prompt = f"""Based on the following list of books and ratings, write a 100-word paragraph that describes what this person's personality and inner world might be like.

{book_list}

• Do not mention any book titles, authors.
• You may mention high level topics. like (politics, palestine, psychology)
• Use "they" as the pronoun.
• Focus on what their reading preferences suggest about their character — such as how they think, what they value, and how they see the world.
• Make the paragraph semantically and emotionally rich, but avoid overly poetic language or complex vocabulary.
• Do not do literary analysis. This is a personality sketch inferred from taste.
• Think of it like reading between the lines of someone's bookshelf to understand who they are."""

    result = make_api_call(prompt, api_key)
    
    if result:
        return result
    else:
        # Fallback description
        return "They are an eclectic reader who enjoys diverse books across multiple genres. Their reading choices suggest curiosity about the world and appreciation for both entertainment and learning. They tend to rate books thoughtfully, showing genuine engagement with the content. Their preferences indicate someone who reads for personal growth, escapism, and intellectual stimulation. They appear to value good storytelling and meaningful themes that resonate with their life experiences and interests."

def main():
    print("📚 Generating taste profiles for 200 book users...")
    
    # Load the generated users
    try:
        with open('book_users_200.json', 'r') as f:
            users = json.load(f)
    except FileNotFoundError:
        print("❌ Error: book_users_200.json not found. Run generate_200_book_users.py first.")
        return
    
    profiles = []
    
    for i, user in enumerate(users):
        print(f"Processing {i+1}/200: {user['name']}")
        
        # Check if this is Yahya - use existing taste profile if available
        if user['name'] == "Yahya Rahhawi":
            try:
                with open('combined_profiles_e5.json', 'r') as f:
                    existing_profiles = json.load(f)
                    for profile in existing_profiles:
                        if profile['name'] == "Yahya Rahhawi":
                            profile_data = {
                                "name": user['name'],
                                "book_taste": profile['book_taste'],
                                "books": user['books'],
                                "personality_archetype": user.get('personality_archetype', ''),
                                "personality_description": user.get('personality_description', '')
                            }
                            profiles.append(profile_data)
                            print(f"  ✓ Used existing taste profile for Yahya ({len(profile['book_taste'].split())} words)")
                            break
                    else:
                        # Fallback: generate new profile if not found
                        taste_profile = generate_taste_profile(user)
                        profile_data = {
                            "name": user['name'],
                            "book_taste": taste_profile,
                            "books": user['books'],
                            "personality_archetype": user.get('personality_archetype', ''),
                            "personality_description": user.get('personality_description', '')
                        }
                        profiles.append(profile_data)
                        print(f"  ✓ Generated new taste profile for Yahya ({len(taste_profile.split())} words)")
            except FileNotFoundError:
                # Generate new profile if existing file not found
                taste_profile = generate_taste_profile(user)
                profile_data = {
                    "name": user['name'],
                    "book_taste": taste_profile,
                    "books": user['books'],
                    "personality_archetype": user.get('personality_archetype', ''),
                    "personality_description": user.get('personality_description', '')
                }
                profiles.append(profile_data)
                print(f"  ✓ Generated new taste profile for Yahya ({len(taste_profile.split())} words)")
        else:
            # Generate new profile for all other users
            taste_profile = generate_taste_profile(user)
            
            profile_data = {
                "name": user['name'],
                "book_taste": taste_profile,
                "books": user['books'],  # Keep original book data
                "personality_archetype": user.get('personality_archetype', ''),
                "personality_description": user.get('personality_description', '')
            }
            
            profiles.append(profile_data)
            print(f"  ✓ Generated taste profile ({len(taste_profile.split())} words)")
    
    # Save profiles
    output_file = "book_taste_profiles_200.json"
    with open(output_file, 'w') as f:
        json.dump(profiles, f, indent=2)
    
    print(f"\n🎉 Generated taste profiles for {len(profiles)} users!")
    print(f"📁 Saved to: {output_file}")

if __name__ == "__main__":
    main()