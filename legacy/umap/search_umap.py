#!/usr/bin/env python3
"""
Search in UMAP space - finds closest matches and visualizes position
"""

import os
import json
import requests
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv

try:
    import umap
except ImportError:
    raise SystemExit("❌ UMAP not found. Install with: pip install umap-learn")

load_dotenv()

class UMAPSearcher:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.original_data = None
        self.umap2d_coords = None
        self.umap3d_coords = None
        self.labels = None
        self.descriptions = None
        self.umap2d_reducer = None
        self.umap3d_reducer = None
        
        if not self.api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in environment variables")
    
    def generate_gemini_embedding(self, text):
        """Generate embedding using configured Gemini embedding model"""
        model = os.getenv('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-001')
        dimensions = int(os.getenv('GEMINI_EMBEDDING_DIMENSIONS', '768'))
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={self.api_key}"
        
        headers = {"Content-Type": "application/json"}
        data = {
            "model": f"models/{model}",
            "content": {"parts": [{"text": text}]},
            "taskType": "SEMANTIC_SIMILARITY",
            "outputDimensionality": dimensions
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            if 'embedding' in result and 'values' in result['embedding']:
                return result['embedding']['values']
            return None
        except Exception as e:
            return None
    
    def load_data(self):
        """Load original data and UMAP coordinates"""
        # Load original data
        try:
            with open('main.json', 'r') as f:
                self.original_data = json.load(f)
            
            self.labels = []
            self.descriptions = []
            
            for entry in self.original_data:
                self.labels.append(entry.get('name', 'Unknown'))
                self.descriptions.append(entry.get('book_taste', 'No description'))
            
            print(f"✅ Loaded {len(self.original_data)} users from main.json")
            
        except FileNotFoundError:
            raise FileNotFoundError("main.json not found")
        
        # Load UMAP coordinates
        try:
            with open('main_UMAP2D.json', 'r') as f:
                umap2d_data = json.load(f)
            self.umap2d_coords = np.array([[user['x'], user['y']] for user in umap2d_data['users']])
            print("✅ Loaded 2D UMAP coordinates")
        except FileNotFoundError:
            print("⚠️ main_UMAP2D.json not found")
        
        try:
            with open('main_UMAP3D.json', 'r') as f:
                umap3d_data = json.load(f)
            self.umap3d_coords = np.array([[user['x'], user['y'], user['z']] for user in umap3d_data['users']])
            print("✅ Loaded 3D UMAP coordinates")
        except FileNotFoundError:
            print("⚠️ main_UMAP3D.json not found")
        
        return self
    
    def fit_umap_models(self):
        """Fit UMAP models to allow new point transformation"""
        # Extract and normalize features
        features = []
        for entry in self.original_data:
            features.append(entry['books_vector'])
        
        X = np.array(features, dtype=np.float32)
        # Unit normalize (as done in generate_UMAP.py)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / (norms + 1e-9)
        
        print("🔧 Fitting UMAP models...")
        
        # Fit 2D UMAP
        if self.umap2d_coords is not None:
            self.umap2d_reducer = umap.UMAP(
                n_components=2,
                n_neighbors=20,
                min_dist=0.1,
                metric="cosine",
                random_state=42
            )
            self.umap2d_reducer.fit(X_norm)
        
        # Fit 3D UMAP  
        if self.umap3d_coords is not None:
            self.umap3d_reducer = umap.UMAP(
                n_components=3,
                n_neighbors=20,
                min_dist=0.1,
                metric="cosine",
                random_state=42
            )
            self.umap3d_reducer.fit(X_norm)
        
        print("✅ UMAP models fitted")
        return self
    
    def find_closest_by_embedding(self, description, k=5):
        """Find closest matches using cosine similarity in embedding space"""
        new_embedding = self.generate_gemini_embedding(description)
        if new_embedding is None:
            return None, None
        
        new_embedding = np.array(new_embedding)
        similarities = []
        
        for entry in self.original_data:
            existing_embedding = np.array(entry['books_vector'])
            # Cosine similarity
            cosine_sim = np.dot(new_embedding, existing_embedding) / (
                np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding)
            )
            similarities.append(cosine_sim)
        
        similarities = np.array(similarities)
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                'name': self.labels[idx],
                'description': self.descriptions[idx],
                'similarity': float(similarities[idx]),
                'rank': i + 1,
                'index': idx
            })
        
        return results, new_embedding
    
    def transform_to_umap_space(self, new_embedding):
        """Transform new embedding to UMAP space"""
        # Normalize the new embedding
        new_embedding = np.array(new_embedding).reshape(1, -1)
        norm = np.linalg.norm(new_embedding)
        new_embedding_norm = new_embedding / (norm + 1e-9)
        
        coords = {}
        
        if self.umap2d_reducer is not None:
            coords_2d = self.umap2d_reducer.transform(new_embedding_norm)
            coords['2d'] = coords_2d[0]
        
        if self.umap3d_reducer is not None:
            coords_3d = self.umap3d_reducer.transform(new_embedding_norm)
            coords['3d'] = coords_3d[0]
        
        return coords
    
    def create_2d_visualization(self, new_coords_2d, description, similar_users):
        """Create 2D UMAP visualization with new point"""
        fig = go.Figure()
        
        # Prepare descriptions for hover (truncate long ones)
        hover_descriptions = []
        for desc in self.descriptions:
            if len(desc) > 150:
                hover_descriptions.append(desc[:150] + "...")
            else:
                hover_descriptions.append(desc)
        
        # Add existing users
        fig.add_trace(go.Scatter(
            x=self.umap2d_coords[:, 0],
            y=self.umap2d_coords[:, 1],
            mode='markers',
            marker=dict(size=6, color='lightblue', opacity=0.7),
            text=self.labels,
            customdata=hover_descriptions,
            name='Existing Users',
            hovertemplate='<b>%{text}</b><br>' +
                         '<i>%{customdata}</i><br>' +
                         'X: %{x:.3f} | Y: %{y:.3f}<extra></extra>'
        ))
        
        # Highlight similar users
        if similar_users:
            similar_indices = [user['index'] for user in similar_users[:3]]
            similar_hover_descriptions = [hover_descriptions[i] for i in similar_indices]
            fig.add_trace(go.Scatter(
                x=self.umap2d_coords[similar_indices, 0],
                y=self.umap2d_coords[similar_indices, 1],
                mode='markers',
                marker=dict(size=10, color='orange', opacity=0.9),
                text=[self.labels[i] for i in similar_indices],
                customdata=similar_hover_descriptions,
                name='Most Similar',
                hovertemplate='<b>%{text}</b><br>' +
                             '<i>%{customdata}</i><br>' +
                             'X: %{x:.3f} | Y: %{y:.3f}<extra></extra>'
            ))
        
        # Add new point
        fig.add_trace(go.Scatter(
            x=[new_coords_2d[0]],
            y=[new_coords_2d[1]],
            mode='markers+text',
            marker=dict(size=15, color='red', symbol='diamond'),
            text=[f"Your Description"],
            textposition="top center",
            name='Your Description',
            hovertemplate=f'<b>Your Description</b><br>{description}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'UMAP 2D: "{description[:50]}..."',
            xaxis_title='UMAP-1',
            yaxis_title='UMAP-2',
            width=1200,
            height=800
        )
        
        return fig
    
    def create_3d_visualization(self, new_coords_3d, description, similar_users):
        """Create 3D UMAP visualization with new point"""
        fig = go.Figure()
        
        # Prepare descriptions for hover (truncate long ones)
        hover_descriptions = []
        for desc in self.descriptions:
            if len(desc) > 150:
                hover_descriptions.append(desc[:150] + "...")
            else:
                hover_descriptions.append(desc)
        
        # Add existing users
        fig.add_trace(go.Scatter3d(
            x=self.umap3d_coords[:, 0],
            y=self.umap3d_coords[:, 1],
            z=self.umap3d_coords[:, 2],
            mode='markers',
            marker=dict(size=5, color='lightblue', opacity=0.7),
            text=self.labels,
            customdata=hover_descriptions,
            name='Existing Users',
            hovertemplate='<b>%{text}</b><br>' +
                         '<i>%{customdata}</i><br>' +
                         'X: %{x:.3f} | Y: %{y:.3f} | Z: %{z:.3f}<extra></extra>'
        ))
        
        # Highlight similar users
        if similar_users:
            similar_indices = [user['index'] for user in similar_users[:3]]
            similar_hover_descriptions = [hover_descriptions[i] for i in similar_indices]
            fig.add_trace(go.Scatter3d(
                x=self.umap3d_coords[similar_indices, 0],
                y=self.umap3d_coords[similar_indices, 1],
                z=self.umap3d_coords[similar_indices, 2],
                mode='markers',
                marker=dict(size=8, color='orange', opacity=0.9),
                text=[self.labels[i] for i in similar_indices],
                customdata=similar_hover_descriptions,
                name='Most Similar',
                hovertemplate='<b>%{text}</b><br>' +
                             '<i>%{customdata}</i><br>' +
                             'X: %{x:.3f} | Y: %{y:.3f} | Z: %{z:.3f}<extra></extra>'
            ))
        
        # Add new point
        fig.add_trace(go.Scatter3d(
            x=[new_coords_3d[0]],
            y=[new_coords_3d[1]], 
            z=[new_coords_3d[2]],
            mode='markers+text',
            marker=dict(size=12, color='red', symbol='diamond'),
            text=[f"Your Description"],
            textposition="top center",
            name='Your Description',
            hovertemplate=f'<b>Your Description</b><br>{description}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: %{{z:.3f}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'UMAP 3D: "{description[:50]}..."',
            scene=dict(
                xaxis_title='UMAP-1',
                yaxis_title='UMAP-2',
                zaxis_title='UMAP-3'
            ),
            width=1200,
            height=800
        )
        
        return fig

def simple_search(description):
    """Simple search - just print closest match"""
    try:
        searcher = UMAPSearcher()
        searcher.load_data()
        
        similar_users, _ = searcher.find_closest_by_embedding(description, k=1)
        
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
    """Interactive search with UMAP visualization"""
    try:
        print("🧭 UMAP Description Search")
        print("="*40)
        
        searcher = UMAPSearcher()
        searcher.load_data()
        searcher.fit_umap_models()
        
        description = input("Enter description: ").strip()
        if not description:
            description = "someone who loves math, science, and the mysteries of the universe"
            print(f"Using default: {description}")
        
        print(f"\n🔍 Searching for: '{description}'")
        
        # Find similar users
        similar_users, new_embedding = searcher.find_closest_by_embedding(description, k=5)
        
        if not similar_users:
            print("❌ No matches found")
            return
        
        print(f"\n🎯 Top matches:")
        for user in similar_users[:3]:
            print(f"   {user['rank']}. {user['name']} (similarity: {user['similarity']:.3f})")
        
        # Transform to UMAP space
        new_coords = searcher.transform_to_umap_space(new_embedding)
        
        # Create visualizations
        if '2d' in new_coords:
            print("\n📊 Creating 2D visualization...")
            fig_2d = searcher.create_2d_visualization(new_coords['2d'], description, similar_users)
            fig_2d.write_html('umap_2d_search_result.html')
            print("💾 2D visualization saved to: umap_2d_search_result.html")
            fig_2d.show()
        
        if '3d' in new_coords:
            print("\n📊 Creating 3D visualization...")
            fig_3d = searcher.create_3d_visualization(new_coords['3d'], description, similar_users)
            fig_3d.write_html('umap_3d_search_result.html')
            print("💾 3D visualization saved to: umap_3d_search_result.html")
            fig_3d.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    # 🎯 EDIT THIS LINE for simple search:
    description = "someone who loves math, science, and the mysteries of the universe"
    
    print("UMAP Search Options:")
    print("1. Simple search (clean output)")
    print("2. Interactive search with visualization")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        interactive_search()
    else:
        simple_search(description)

if __name__ == "__main__":
    main()