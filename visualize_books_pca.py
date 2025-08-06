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

def randomize_names_in_books_json(input_file, output_file):
    """Efficiently process large JSON file and randomize names, ensuring Yahya Rahhawi is included"""
    print("Generating 199 unique random names...")
    new_names = generate_unique_names(199)
    # Add Yahya Rahhawi as one of the names
    new_names.append("Yahya Rahhawi")
    random.shuffle(new_names)  # Shuffle to randomize position
    
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

def extract_books_features(data):
    """Extract books_vector features from the data for PCA"""
    features = []
    labels = []
    
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
    
    # Extract features focusing only on books_vector
    for entry in data_entries:
        name = entry.get('name', entry.get('Name', 'Unknown'))
        labels.append(name)
        
        # Add books vector
        if 'books_vector' in entry and isinstance(entry['books_vector'], list):
            features.append(entry['books_vector'])
        else:
            print(f"Warning: No books_vector found for {name}")
            # Add zero vector if missing
            features.append([0.0] * 1024)  # Assuming 1024 dimensions
    
    return np.array(features), labels

def create_books_pca_visualization(data):
    """Create 3D PCA visualization using plotly, highlighting Yahya Rahhawi in red"""
    print("Extracting books vector features...")
    features, labels = extract_books_features(data)
    
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
    
    # Create colors array - red for Yahya Rahhawi, blue for others
    colors = []
    for label in labels:
        if label == "Yahya Rahhawi":
            colors.append("red")
        else:
            colors.append("lightblue")
    
    # Create 3D scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        z=pca_result[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color=colors,
            opacity=0.8
        ),
        text=labels,
        hovertemplate='<b>%{text}</b><br>' +
                      f'PC1: %{{x:.3f}}<br>' +
                      f'PC2: %{{y:.3f}}<br>' +
                      f'PC3: %{{z:.3f}}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'3D PCA Visualization of Books Embeddings (1024D → 3D)<br>200 People - Yahya Rahhawi highlighted in red<br>Explained Variance: {sum(pca.explained_variance_ratio_):.1%}',
        scene=dict(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
            zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%})'
        ),
        width=1000,
        height=800
    )
    
    # Save the plot
    output_html = "/Users/yahyarahhawi/Developer/Mello/Mello-ML/books_pca_3d_yahya_highlighted.html"
    fig.write_html(output_html)
    print(f"3D PCA visualization saved to: {output_html}")
    
    # Also show the plot
    fig.show()
    
    return fig

def main():
    input_file = "/Users/yahyarahhawi/Developer/Mello/Mello-ML/book_embeddings_e5_200.json"
    output_file = "/Users/yahyarahhawi/Developer/Mello/Mello-ML/book_embeddings_e5_200_randomized.json"
    
    print("Step 1: Randomizing names (including Yahya Rahhawi)...")
    updated_data = randomize_names_in_books_json(input_file, output_file)
    
    print("\nStep 2: Creating PCA visualization...")
    create_books_pca_visualization(updated_data)
    
    print("\nProcess completed!")

if __name__ == "__main__":
    main() 