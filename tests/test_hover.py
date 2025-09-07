#!/usr/bin/env python3
"""
Test hover behavior in visualizer
"""

import numpy as np
from user import User
from population import Population
from visualizer import Visualizer

# Create test population
population = Population("Test Population")

# Add a test user with bio
user = User("Test User Name")
user.profile_data = {"bio": "This bio should NOT appear in hover tooltips"}
user.interests_embedding = np.random.rand(768)
user.openness_embedding = np.random.rand(768)
user.conscientiousness_embedding = np.random.rand(768)
user.extraversion_embedding = np.random.rand(768)
user.agreeableness_embedding = np.random.rand(768)
user.neuroticism_embedding = np.random.rand(768)
population.add_user(user)

# Add more test users
for i in range(5):
    user_i = User(f"User {i+2}")
    user_i.profile_data = {"bio": f"Bio {i+2} that should not show in hover"}
    user_i.interests_embedding = np.random.rand(768)
    user_i.openness_embedding = np.random.rand(768)
    user_i.conscientiousness_embedding = np.random.rand(768)
    user_i.extraversion_embedding = np.random.rand(768)
    user_i.agreeableness_embedding = np.random.rand(768)
    user_i.neuroticism_embedding = np.random.rand(768)
    population.add_user(user_i)

# Test visualization
visualizer = Visualizer()

print("Testing hover templates...")
print("Creating PCA plot - check hover shows ONLY names, not bio")

try:
    fig = visualizer.plot_population_pca(population, mode='combined')
    
    # Inspect the figure traces
    for i, trace in enumerate(fig.data):
        print(f"\nTrace {i}:")
        print(f"  Name: {trace.name}")
        if hasattr(trace, 'text'):
            print(f"  Text: {trace.text}")
        if hasattr(trace, 'hovertext'):
            print(f"  Hovertext: {trace.hovertext}")
        if hasattr(trace, 'hovertemplate'):
            print(f"  Hovertemplate: {trace.hovertemplate}")
    
    # Save as HTML to test
    fig.write_html("test_hover.html")
    print(f"\n✅ Test plot saved as test_hover.html")
    print("Open this file in a browser and check hover tooltips")
    
except Exception as e:
    print(f"❌ Error: {e}")