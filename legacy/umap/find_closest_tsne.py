#!/usr/bin/env python3
"""
Find closest person using t-SNE transformation
"""

import os
import json
import requests
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

class TSNEMatcher:
    def __init__(self):
        self.scaler = None
        self.tsne_coords = None
        self.labels = None
        self.descriptions = None
        self.api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in environment variables")
    
    def generate_gemini_embedding(self, text):
        """Generate embedding using Google Gemini embedding model"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={self.api_key}"
        
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": text}]},
            "taskType": "SEMANTIC_SIMILARITY"
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            if 'embedding' in result and 'values' in result['embedding']:
                return result['embedding']['values']
            else:
                return None
                
        except Exception as e:
            return None
    
    def load_existing_data(self):
        """Load and fit t-SNE model from existing data"""
        # Load main.json
        try:
            with open('main.json', 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("main.json not found")
        
        # Extract features and info
        features = []
        self.labels = []
        self.descriptions = []
        
        for entry in data:
            self.labels.append(entry.get('name', 'Unknown'))
            self.descriptions.append(entry.get('book_taste', 'No description'))
            features.append(entry['books_vector'])
        
        features = np.array(features)
        
        # Check if t-SNE results already exist
        try:
            with open('main_TSNE.json', 'r') as f:
                tsne_data = json.load(f)
            
            # Extract coordinates
            self.tsne_coords = np.array([
                [user['x'], user['y'], user['z']] 
                for user in tsne_data['users']
            ])
            
            # Fit scaler for new transformations
            self.scaler = StandardScaler()
            self.scaler.fit(features)
            
            print("✅ Loaded existing t-SNE coordinates from main_TSNE.json")
            
        except FileNotFoundError:
            print("⚠️ main_TSNE.json not found, generating t-SNE transformation...")
            self.fit_tsne(features)
        
        return self
    
    def fit_tsne(self, features):
        """Fit t-SNE transformation"""
        print("🔧 Standardizing features...")
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        print("🎯 Applying t-SNE (this may take a few minutes)...")
        tsne = TSNE(
            n_components=3,
            perplexity=30,
            learning_rate=200,
            n_iter=1000,
            random_state=42,
            verbose=0
        )
        
        self.tsne_coords = tsne.fit_transform(features_scaled)
        
        # Save results
        self.save_tsne_results()
        print("✅ t-SNE transformation complete!")
    
    def save_tsne_results(self):
        """Save t-SNE results to file"""
        tsne_data = {
            "users": [
                {
                    "name": self.labels[i],
                    "x": float(self.tsne_coords[i, 0]),
                    "y": float(self.tsne_coords[i, 1]),
                    "z": float(self.tsne_coords[i, 2]),
                    "isCurrentUser": False,
                    "special_user": False
                }
                for i in range(len(self.labels))
            ]
        }
        
        with open('main_TSNE.json', 'w') as f:
            json.dump(tsne_data, f, indent=2)
    
    def find_similar_in_embedding_space(self, description, k=5):
        """Find similar users by comparing embeddings directly"""
        # Get embedding for new description
        new_embedding = self.generate_gemini_embedding(description)
        if new_embedding is None:
            return None, None
        
        # Load original embeddings
        with open('main.json', 'r') as f:
            data = json.load(f)
        
        # Calculate cosine similarities
        new_embedding = np.array(new_embedding)
        similarities = []
        
        for i, entry in enumerate(data):
            existing_embedding = np.array(entry['books_vector'])
            # Cosine similarity
            cosine_sim = np.dot(new_embedding, existing_embedding) / (
                np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding)
            )
            similarities.append(cosine_sim)
        
        # Find most similar
        similarities = np.array(similarities)
        most_similar_indices = np.argsort(similarities)[::-1][:k]
        
        similar_users = []
        for i, idx in enumerate(most_similar_indices):
            similar_users.append({
                'name': self.labels[idx],
                'description': self.descriptions[idx],
                'similarity': float(similarities[idx]),
                'rank': i + 1
            })
        
        return similar_users, new_embedding
    
    def project_to_tsne_space(self, new_embedding):
        """Project new embedding to existing t-SNE space (approximation)"""
        # Standardize the new embedding
        new_embedding_scaled = self.scaler.transform(new_embedding.reshape(1, -1))
        
        # Find closest point in original space and use its t-SNE coordinates as approximation
        # This is a simplification - true t-SNE projection requires refitting
        with open('main.json', 'r') as f:
            data = json.load(f)
        
        original_embeddings = np.array([entry['books_vector'] for entry in data])
        original_scaled = self.scaler.transform(original_embeddings)
        
        # Find nearest neighbor in scaled space
        distances = np.linalg.norm(original_scaled - new_embedding_scaled, axis=1)
        nearest_idx = np.argmin(distances)
        
        # Use the t-SNE coordinates of the nearest neighbor as approximation
        approximate_tsne_coords = self.tsne_coords[nearest_idx]
        
        return approximate_tsne_coords
    
    def create_visualization(self, new_coords, description, similar_users):
        """Create 3D visualization with new point"""
        fig = go.Figure()
        
        # Add existing users
        fig.add_trace(go.Scatter3d(
            x=self.tsne_coords[:, 0],
            y=self.tsne_coords[:, 1],
            z=self.tsne_coords[:, 2],
            mode='markers',
            marker=dict(size=6, color='lightblue', opacity=0.7),
            text=self.labels,
            name='Existing Users',
            hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
        ))
        
        # Highlight similar users
        if similar_users:
            similar_indices = [self.labels.index(user['name']) for user in similar_users[:3]]
            fig.add_trace(go.Scatter3d(
                x=self.tsne_coords[similar_indices, 0],
                y=self.tsne_coords[similar_indices, 1],
                z=self.tsne_coords[similar_indices, 2],
                mode='markers',
                marker=dict(size=10, color='orange', opacity=0.9),
                text=[self.labels[i] for i in similar_indices],
                name='Most Similar',
                hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            ))
        
        # Add new point
        fig.add_trace(go.Scatter3d(
            x=[new_coords[0]],
            y=[new_coords[1]],
            z=[new_coords[2]],
            mode='markers+text',
            marker=dict(size=15, color='red', symbol='diamond'),
            text=[f"New: {description[:20]}..."],
            textposition="top center",
            name='Your Description',
            hovertemplate=f'<b>Your Description</b><br>{description}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: %{{z:.3f}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f't-SNE Space: "{description}"',
            scene=dict(
                xaxis_title='t-SNE Component 1',
                yaxis_title='t-SNE Component 2',
                zaxis_title='t-SNE Component 3'
            ),
            width=1200,
            height=800
        )
        
        return fig

def find_closest_match(description):
    """Simple function to find closest match"""
    try:
        matcher = TSNEMatcher()
        matcher.load_existing_data()
        
        # Find similar users
        similar_users, new_embedding = matcher.find_similar_in_embedding_space(description, k=1)
        
        if similar_users:
            closest = similar_users[0]
            print(f"🎯 Closest match: {closest['name']}")
            print(f"📖 Their description: {closest['description']}")
            print(f"📊 Similarity: {closest['similarity']:.4f}")
        else:
            print("❌ No matches found")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def interactive_search():
    """Interactive search with visualization"""
    try:
        print("🧬 t-SNE Description Matcher")
        print("="*40)
        
        matcher = TSNEMatcher()
        matcher.load_existing_data()
        
        description = input("Enter description: ").strip()
        if not description:
            description = "someone who loves math, science, and the mysteries of the universe"
            print(f"Using default: {description}")
        
        print(f"\n🔍 Searching for: '{description}'")
        
        # Find similar users
        similar_users, new_embedding = matcher.find_similar_in_embedding_space(description, k=5)
        
        if similar_users:
            print(f"\n🎯 Top matches:")
            for user in similar_users[:3]:
                print(f"   {user['rank']}. {user['name']} (similarity: {user['similarity']:.3f})")
            
            # Project to t-SNE space and visualize
            new_coords = matcher.project_to_tsne_space(new_embedding)
            
            fig = matcher.create_visualization(new_coords, description, similar_users)
            fig.write_html('tsne_search_result.html')
            print(f"\n📊 Visualization saved to: tsne_search_result.html")
            fig.show()
        else:
            print("❌ No matches found")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    # 🎯 EDIT THIS LINE for simple mode:
    description = "someone who loves math, science, and the mysteries of the universe"
    
    print("Choose mode:")
    print("1. Simple search (edit description in code)")
    print("2. Interactive search with visualization")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        interactive_search()
    else:
        find_closest_match(description)

if __name__ == "__main__":
    main()