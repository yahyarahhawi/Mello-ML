import os
import json
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_taste_profile_for_user(user_data):
    """Generate taste profiles for real user using same prompts as synthetic users"""
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    # Generate book taste description
    books = user_data.get('books', [])
    book_list = ""
    for book in books:
        book_list += f"• {book['title']} by {book['author']} - {book['rating']} stars\n"
    
    book_prompt = f"""Based on the following list of books and ratings, write a 100-word paragraph that describes what this person's personality and inner world might be like.

{book_list}

• Do not mention any book titles, authors.
• You may mention high level topics. like (politics, palestine, psychology)
• Use "they" as the pronoun.
• Focus on what their reading preferences suggest about their character — such as how they think, what they value, and how they see the world.
• Make the paragraph semantically and emotionally rich, but avoid overly poetic language or complex vocabulary.
• Do not do literary analysis. This is a personality sketch inferred from taste.
• Think of it like reading between the lines of someone's bookshelf to understand who they are."""

    # Generate movie taste description
    movies = user_data.get('movies', [])
    movie_list = ""
    for movie in movies:
        movie_list += f"• {movie['title']} ({movie['year']}) - {movie['rating']} stars\n"
    
    movie_prompt = f"""Based on the following list of movies and ratings, write a 100-word paragraph that describes what this person's personality and inner world might be like.

{movie_list}

• Do not mention any movie titles, directors, or actors.
• You may mention high level themes, genres, or topics.
• Use "they" as the pronoun.
• Focus on what their viewing preferences suggest about their character — such as how they think, what they value, and how they see the world.
• Make the paragraph semantically and emotionally rich, but avoid overly poetic language or complex vocabulary.
• Do not do film analysis. This is a personality sketch inferred from taste.
• Think of it like reading between the lines of someone's movie collection to understand who they are."""

    # Generate music taste description
    songs = user_data.get('songs', [])
    song_list = ""
    for song in songs:
        song_list += f"• {song['title']} by {song['artist']}\n"
    
    music_prompt = f"""Based on the following list of songs, write a 100-word paragraph that describes what this person's personality and inner world might be like.

{song_list}

• Do not mention any song titles, artists, or albums.
• You may mention high level musical genres, themes, or moods.
• Use "they" as the pronoun.
• Focus on what their musical preferences suggest about their character — such as how they think, what they value, and how they see the world.
• Make the paragraph semantically and emotionally rich, but avoid overly poetic language or complex vocabulary.
• Do not do musical analysis. This is a personality sketch inferred from taste.
• Think of it like reading between the lines of someone's playlist to understand who they are."""

    # Make API calls
    book_taste = make_api_call(book_prompt, api_key)
    movie_taste = make_api_call(movie_prompt, api_key)
    music_taste = make_api_call(music_prompt, api_key)
    
    return {
        "name": user_data.get('name'),
        "book_taste": book_taste,
        "movie_taste": movie_taste,
        "music_taste": music_taste
    }

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

def generate_e5_embeddings(taste_profile):
    """Generate E5 embeddings for the taste profile"""
    
    print("Loading multilingual-e5-large model...")
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    
    # Add prefixes for E5 model and generate embeddings
    book_text = f"query: {taste_profile['book_taste']}"
    movie_text = f"query: {taste_profile['movie_taste']}"
    music_text = f"query: {taste_profile['music_taste']}"
    
    book_embedding = model.encode(book_text, convert_to_tensor=True, normalize_embeddings=False)
    movie_embedding = model.encode(movie_text, convert_to_tensor=True, normalize_embeddings=False)
    music_embedding = model.encode(music_text, convert_to_tensor=True, normalize_embeddings=False)
    
    return {
        "name": taste_profile['name'],
        "books_vector": book_embedding.cpu().numpy().tolist(),
        "movies_vector": movie_embedding.cpu().numpy().tolist(),
        "music_vector": music_embedding.cpu().numpy().tolist(),
        "book_taste": taste_profile['book_taste'],
        "movie_taste": taste_profile['movie_taste'],
        "music_taste": taste_profile['music_taste']
    }

def plot_with_existing_users(my_embeddings):
    """Plot real user with existing synthetic users using E5 embeddings"""
    
    # Load existing E5 embeddings
    try:
        with open('semantic_profiles_e5.json', 'r', encoding='utf-8') as f:
            existing_profiles = json.load(f)
    except FileNotFoundError:
        print("Error: semantic_profiles_e5.json not found. Run generate_e5_embeddings.py first.")
        return
    
    categories = ['books', 'movies', 'music']
    
    for category in categories:
        print(f"\nPlotting {category} taste comparison...")
        
        # Extract embeddings for this category
        vector_key = f"{category}_vector"
        
        # Existing users
        existing_embeddings = []
        existing_names = []
        
        for profile in existing_profiles:
            if vector_key in profile:
                existing_embeddings.append(profile[vector_key])
                existing_names.append(profile['name'])
        
        # Add real user
        if vector_key in my_embeddings:
            all_embeddings = existing_embeddings + [my_embeddings[vector_key]]
            all_names = existing_names + [my_embeddings['name']]
        else:
            print(f"No {category} embedding found for real user")
            continue
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings)
        
        # Apply PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_array)
        
        # Create plot
        plt.figure(figsize=(14, 10))
        
        # Plot synthetic users in blue
        plt.scatter(embeddings_2d[:-1, 0], embeddings_2d[:-1, 1], 
                   alpha=0.7, c='lightblue', s=50, label='Synthetic Users')
        
        # Plot real user in red
        plt.scatter(embeddings_2d[-1, 0], embeddings_2d[-1, 1], 
                   alpha=1.0, c='red', s=200, marker='*', label='Yahya Rahhawi (Real User)')
        
        # Add labels for all points
        for i, name in enumerate(all_names):
            if i == len(all_names) - 1:  # Real user
                plt.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                            xytext=(10, 10), textcoords='offset points', 
                            fontsize=10, fontweight='bold', color='red')
            else:  # Synthetic users
                plt.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                            xytext=(5, 5), textcoords='offset points', 
                            fontsize=8, alpha=0.7)
        
        plt.title(f'{category.title()} Taste: Real User vs Synthetic Users (E5 Embeddings)')
        plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        filename = f'real_vs_synthetic_{category}_e5.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {filename}")
        plt.close()

def save_combined_embeddings(my_embeddings):
    """Save combined embeddings (real + synthetic) for similarity testing"""
    
    # Load existing embeddings
    try:
        with open('semantic_profiles_e5.json', 'r', encoding='utf-8') as f:
            existing_profiles = json.load(f)
    except FileNotFoundError:
        print("Warning: Could not load existing E5 embeddings")
        existing_profiles = []
    
    # Combine with real user
    combined_profiles = existing_profiles + [my_embeddings]
    
    # Save combined profiles
    with open('combined_profiles_e5.json', 'w', encoding='utf-8') as f:
        json.dump(combined_profiles, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(combined_profiles)} profiles (including real user) to combined_profiles_e5.json")

def main():
    """Main function to process real user profile"""
    
    # Load real user data
    try:
        with open('raw_real_users.json', 'r', encoding='utf-8') as f:
            real_users = json.load(f)
        
        if not real_users:
            print("No real users found in raw_real_users.json")
            return
        
        real_user = real_users[0]  # Yahya's profile
    except FileNotFoundError:
        print("Error: raw_real_users.json not found")
        return
    
    print(f"Processing profile for: {real_user.get('name', 'Unknown')}")
    
    # Generate taste profile
    print("\nGenerating taste descriptions...")
    taste_profile = generate_taste_profile_for_user(real_user)
    
    if not all([taste_profile['book_taste'], taste_profile['movie_taste'], taste_profile['music_taste']]):
        print("Failed to generate complete taste profile")
        return
    
    print("✓ Generated taste descriptions:")
    print(f"  Book taste: {taste_profile['book_taste'][:50]}...")
    print(f"  Movie taste: {taste_profile['movie_taste'][:50]}...")
    print(f"  Music taste: {taste_profile['music_taste'][:50]}...")
    
    # Generate embeddings
    print("\nGenerating E5 embeddings...")
    my_embeddings = generate_e5_embeddings(taste_profile)
    print("✓ Generated E5 embeddings")
    
    # Plot with existing users
    print("\nCreating comparison plots...")
    plot_with_existing_users(my_embeddings)
    
    # Save combined embeddings
    print("\nSaving combined embeddings...")
    save_combined_embeddings(my_embeddings)
    
    print(f"\n🎉 Successfully processed {real_user['name']}'s profile!")
    print("Ready for similarity matching against synthetic users!")

if __name__ == "__main__":
    main()