import json
import random
from faker import Faker
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

def generate_unique_names(count=200):
    """Generate 200 unique random names"""
    fake = Faker()
    names = set()
    
    while len(names) < count:
        name = fake.name()
        names.add(name)
    
    return list(names)

def randomize_names_in_json(input_file, output_file):
    """Efficiently process large JSON file and randomize names"""
    print("Generating 200 unique random names...")
    new_names = generate_unique_names(200)
    
    print("Processing JSON file...")
    
    # Read the file as a string and process it
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse the JSON
    data = json.loads(content)
    
    # Check if it's a list or dict and update names accordingly
    if isinstance(data, list):
        print(f"Found {len(data)} entries in the list")
        for i, entry in enumerate(data):
            if i < len(new_names):
                if 'name' in entry:
                    entry['name'] = new_names[i]
                elif 'Name' in entry:
                    entry['Name'] = new_names[i]
                # Add other possible name field variations
                for key in entry.keys():
                    if key.lower() == 'name':
                        entry[key] = new_names[i]
                        break
    elif isinstance(data, dict):
        print("Processing dictionary structure...")
        # If it's a dict, try to find the data array
        for key, value in data.items():
            if isinstance(value, list) and len(value) == 200:
                print(f"Found data array with key '{key}'")
                for i, entry in enumerate(value):
                    if i < len(new_names):
                        if 'name' in entry:
                            entry['name'] = new_names[i]
                        elif 'Name' in entry:
                            entry['Name'] = new_names[i]
                        # Add other possible name field variations
                        for entry_key in entry.keys():
                            if entry_key.lower() == 'name':
                                entry[entry_key] = new_names[i]
                                break
                break
    
    # Write the updated data back
    print(f"Writing updated data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("Names randomized successfully!")
    return data

def extract_numerical_features(data):
    """Extract numerical features from the data for PCA with consistent vector lengths"""
    features = []
    labels = []
    
    # First, determine the expected vector lengths by examining all entries
    books_vector_length = None
    movies_vector_length = None
    music_vector_length = None
    
    # Analyze the data structure
    if isinstance(data, list):
        data_entries = data
    elif isinstance(data, dict):
        # Find the main data array
        data_entries = None
        for key, value in data.items():
            if isinstance(value, list):
                data_entries = value
                break
        if data_entries is None:
            print("Could not find data entries!")
            return np.array([]), []
    
    # Determine vector lengths from entries that have them
    for entry in data_entries:
        if 'books_vector' in entry and isinstance(entry['books_vector'], list):
            if books_vector_length is None:
                books_vector_length = len(entry['books_vector'])
        if 'movies_vector' in entry and isinstance(entry['movies_vector'], list):
            if movies_vector_length is None:
                movies_vector_length = len(entry['movies_vector'])
        if 'music_vector' in entry and isinstance(entry['music_vector'], list):
            if music_vector_length is None:
                music_vector_length = len(entry['music_vector'])
    
    print(f"Detected vector lengths - Books: {books_vector_length}, Movies: {movies_vector_length}, Music: {music_vector_length}")
    
    # Extract features with consistent padding
    for entry in data_entries:
        feature_vector = []
        name = entry.get('name', entry.get('Name', 'Unknown'))
        labels.append(name)
        
        # Add books vector (pad with zeros if missing)
        if 'books_vector' in entry and isinstance(entry['books_vector'], list):
            feature_vector.extend(entry['books_vector'])
        elif books_vector_length is not None:
            feature_vector.extend([0.0] * books_vector_length)
        
        # Add movies vector (pad with zeros if missing)
        if 'movies_vector' in entry and isinstance(entry['movies_vector'], list):
            feature_vector.extend(entry['movies_vector'])
        elif movies_vector_length is not None:
            feature_vector.extend([0.0] * movies_vector_length)
        
        # Add music vector (pad with zeros if missing)
        if 'music_vector' in entry and isinstance(entry['music_vector'], list):
            feature_vector.extend(entry['music_vector'])
        elif music_vector_length is not None:
            feature_vector.extend([0.0] * music_vector_length)
        
        if feature_vector:
            features.append(feature_vector)
    
    return np.array(features), labels

def create_pca_visualization(data):
    """Create 3D PCA visualization using plotly"""
    print("Extracting numerical features...")
    features, labels = extract_numerical_features(data)
    
    if len(features) == 0:
        print("No numerical features found for PCA!")
        return
    
    print(f"Found {len(features)} samples with {features.shape[1]} features each")
    
    # Standardize the features
    print("Standardizing features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA
    print("Applying PCA...")
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_scaled)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        z=pca_result[:, 2],
        hover_name=labels,
        title=f'3D PCA Visualization of 200 People with Randomized Names<br>Explained Variance: {sum(pca.explained_variance_ratio_):.1%}',
        labels={
            'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
            'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
            'z': f'PC3 ({pca.explained_variance_ratio_[2]:.1%})'
        }
    )
    
    fig.update_traces(marker=dict(size=5, opacity=0.7, color='lightblue'))
    fig.update_layout(
        scene=dict(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
            zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%})'
        ),
        width=1000,
        height=800
    )
    
    # Save the plot
    output_html = "/Users/yahyarahhawi/Developer/Mello/Mello-ML/pca_3d_randomized_names.html"
    fig.write_html(output_html)
    print(f"3D PCA visualization saved to: {output_html}")
    
    # Also show the plot
    fig.show()
    
    return fig

def main():
    input_file = "/Users/yahyarahhawi/Developer/Mello/Mello-ML/combined_profiles_e5_200.json"
    output_file = "/Users/yahyarahhawi/Developer/Mello/Mello-ML/combined_profiles_e5_200_randomized.json"
    
    print("Step 1: Randomizing names...")
    updated_data = randomize_names_in_json(input_file, output_file)
    
    print("\nStep 2: Creating PCA visualization...")
    create_pca_visualization(updated_data)
    
    print("\nProcess completed!")

if __name__ == "__main__":
    main() 