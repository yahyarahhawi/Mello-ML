#!/usr/bin/env python3
"""
Robust embedding regeneration with rate limiting, retry logic, and resume capability
"""

import os
import json
import requests
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from dotenv import load_dotenv
import time
import random
from pathlib import Path

try:
    import umap
except ImportError:
    print("⚠️ UMAP not found. Install with: pip install umap-learn")
    umap = None

# Load environment variables
load_dotenv()

class RobustEmbeddingGenerator:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model = os.getenv('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-001')
        self.progress_file = 'embedding_progress.json'
        self.output_file = 'main_v2.json'
        
        if not self.api_key:
            raise ValueError("❌ GEMINI_API_KEY not found")
    
    def generate_gemini_embedding(self, text, max_retries=3):
        """Generate embedding with retry logic and exponential backoff"""
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent?key={self.api_key}"
        
        headers = {"Content-Type": "application/json"}
        data = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "taskType": "SEMANTIC_SIMILARITY",
            "outputDimensionality": 768
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=30)
                
                if response.status_code == 429:
                    # Rate limit hit - wait longer
                    wait_time = (2 ** attempt) * 5 + random.uniform(1, 5)  # 5-10s, 10-15s, 20-25s
                    print(f"    ⏳ Rate limit hit, waiting {wait_time:.1f}s (attempt {attempt + 1})")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                if 'embedding' in result and 'values' in result['embedding']:
                    return result['embedding']['values']
                else:
                    print(f"    ⚠️ Unexpected response format")
                    return None
                    
            except requests.exceptions.Timeout:
                wait_time = (2 ** attempt) * 2
                print(f"    ⏳ Timeout, retrying in {wait_time}s (attempt {attempt + 1})")
                time.sleep(wait_time)
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"    ❌ Failed after {max_retries} attempts: {e}")
                    return None
                
                wait_time = (2 ** attempt) * 3
                print(f"    ⏳ Error, retrying in {wait_time}s (attempt {attempt + 1})")
                time.sleep(wait_time)
                
            except Exception as e:
                print(f"    ❌ Unexpected error: {e}")
                return None
        
        return None
    
    def load_progress(self):
        """Load progress from previous run"""
        if Path(self.progress_file).exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                print(f"📁 Resuming from previous run - {len(progress['completed'])} embeddings already done")
                return progress
            except:
                print("⚠️ Could not load progress file, starting fresh")
        
        return {'completed': [], 'failed': []}
    
    def save_progress(self, progress):
        """Save current progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def load_taste_profiles(self):
        """Load the taste profiles"""
        try:
            with open('book_taste_profiles_200.json', 'r') as f:
                profiles = json.load(f)
            print(f"✅ Loaded {len(profiles)} profiles")
            return profiles
        except FileNotFoundError:
            print("❌ book_taste_profiles_200.json not found")
            return None
    
    def regenerate_embeddings(self):
        """Main embedding generation with resume capability"""
        print(f"🧠 Generating embeddings using {self.model}")
        print("🔄 Features: Rate limiting, retry logic, resume capability")
        print("="*60)
        
        # Load data and progress
        profiles = self.load_taste_profiles()
        if profiles is None:
            return None
        
        progress = self.load_progress()
        completed_names = {item['name'] for item in progress['completed']}
        
        # Filter out already completed profiles
        remaining_profiles = [p for p in profiles if p['name'] not in completed_names]
        
        print(f"📊 Total profiles: {len(profiles)}")
        print(f"✅ Already completed: {len(completed_names)}")
        print(f"🔄 Remaining: {len(remaining_profiles)}")
        
        if len(remaining_profiles) == 0:
            print("🎉 All embeddings already generated!")
            return progress['completed']
        
        # Process remaining profiles
        initial_completed_count = len(progress['completed'])
        for i, profile in enumerate(remaining_profiles):
            current_num = initial_completed_count + i + 1
            print(f"\nProcessing {current_num}/{len(profiles)}: {profile['name']}")
            
            # Random delay to be gentle on API (0.5-2.0 seconds)
            delay = random.uniform(0.5, 2.0)
            time.sleep(delay)
            
            embedding = self.generate_gemini_embedding(profile['book_taste'])
            
            if embedding:
                embedding_data = {
                    "name": profile['name'],
                    "book_taste": profile['book_taste'],
                    "books_vector": embedding,
                    "books": profile.get('books', []),
                    "personality_archetype": profile.get('personality_archetype', ''),
                    "personality_description": profile.get('personality_description', ''),
                    "embedding_model": self.model
                }
                
                progress['completed'].append(embedding_data)
                print(f"  ✓ Generated {len(embedding)} dimensional embedding")
                
                # Save progress every 5 successful embeddings
                if len(progress['completed']) % 5 == 0:
                    self.save_progress(progress)
                    print(f"  💾 Progress saved ({len(progress['completed'])} completed)")
                
            else:
                progress['failed'].append(profile['name'])
                print(f"  ✗ Failed to generate embedding")
        
        # Final save
        self.save_progress(progress)
        
        # Save final embeddings file
        with open(self.output_file, 'w') as f:
            json.dump(progress['completed'], f, indent=2)
        
        print(f"\n🎉 Generation complete!")
        print(f"✅ Successful: {len(progress['completed'])}")
        print(f"❌ Failed: {len(progress['failed'])}")
        print(f"📁 Saved to: {self.output_file}")
        
        if progress['failed']:
            print(f"⚠️ Failed profiles: {', '.join(progress['failed'])}")
        
        return progress['completed']
    
    def generate_coordinates(self, data):
        """Generate PCA and UMAP coordinates"""
        if not data:
            print("❌ No embedding data to process")
            return
        
        print(f"\n🔧 Generating coordinates from {len(data)} embeddings...")
        
        features = np.array([entry['books_vector'] for entry in data])
        labels = [entry['name'] for entry in data]
        
        print(f"📊 Feature matrix shape: {features.shape}")
        
        # PCA
        print("📈 Generating PCA coordinates...")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(features_scaled)
        
        pca_data = {
            "users": [
                {
                    "name": labels[i],
                    "x": float(pca_result[i, 0]),
                    "y": float(pca_result[i, 1]),
                    "z": float(pca_result[i, 2]),
                    "isCurrentUser": False,
                    "special_user": False
                }
                for i in range(len(labels))
            ],
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "total_explained_variance": float(sum(pca.explained_variance_ratio_))
        }
        
        with open('main_PCA_v2.json', 'w') as f:
            json.dump(pca_data, f, indent=2)
        
        print(f"✅ PCA saved (explained variance: {sum(pca.explained_variance_ratio_):.1%})")
        
        # UMAP
        if umap is not None:
            print("🧭 Generating UMAP coordinates...")
            
            # Unit normalize
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features_norm = features / (norms + 1e-9)
            
            # 2D UMAP
            umap_2d = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, 
                               metric="cosine", random_state=42, verbose=False)
            coords_2d = umap_2d.fit_transform(features_norm)
            
            umap_2d_data = {
                "users": [
                    {
                        "name": labels[i],
                        "x": float(coords_2d[i, 0]),
                        "y": float(coords_2d[i, 1]),
                        "isCurrentUser": False,
                        "special_user": False
                    }
                    for i in range(len(labels))
                ]
            }
            
            with open('main_UMAP2D_v2.json', 'w') as f:
                json.dump(umap_2d_data, f, indent=2)
            
            # 3D UMAP
            umap_3d = umap.UMAP(n_components=3, n_neighbors=20, min_dist=0.1,
                               metric="cosine", random_state=42, verbose=False)
            coords_3d = umap_3d.fit_transform(features_norm)
            
            umap_3d_data = {
                "users": [
                    {
                        "name": labels[i],
                        "x": float(coords_3d[i, 0]),
                        "y": float(coords_3d[i, 1]),
                        "z": float(coords_3d[i, 2]),
                        "isCurrentUser": False,
                        "special_user": False
                    }
                    for i in range(len(labels))
                ]
            }
            
            with open('main_UMAP3D_v2.json', 'w') as f:
                json.dump(umap_3d_data, f, indent=2)
            
            print("✅ UMAP 2D and 3D saved")
        else:
            print("⚠️ UMAP skipped (not installed)")

def main():
    print("🔄 Robust Embedding Regeneration")
    print("Features: Rate limiting, retry logic, progress saving")
    print("="*60)
    
    try:
        generator = RobustEmbeddingGenerator()
        
        # Generate embeddings
        data = generator.regenerate_embeddings()
        
        if data:
            # Generate coordinates
            generator.generate_coordinates(data)
            
            print("\n✅ All done!")
            print("📁 Files created:")
            print("   - main_v2.json")
            print("   - main_PCA_v2.json") 
            print("   - main_UMAP2D_v2.json")
            print("   - main_UMAP3D_v2.json")
            print("   - embedding_progress.json (can be deleted)")
            
    except KeyboardInterrupt:
        print("\n\n⏸️ Interrupted by user")
        print("💾 Progress has been saved - you can resume by running this script again")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()