#!/usr/bin/env python3
"""
Utility functions for the dual-vector system.
Provides convenient methods for common workflows.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from user import User
from population import Population
from profile_generator import ProfileGenerator
from embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


def process_frontend_json_to_user(json_data: Dict[str, Any], 
                                profile_generator: ProfileGenerator,
                                embedding_generator: EmbeddingGenerator,
                                generate_profiles: bool = True,
                                generate_embeddings: bool = True) -> User:
    """
    Complete workflow: JSON -> User -> Profiles -> Embeddings
    
    Args:
        json_data: Frontend JSON data
        profile_generator: ProfileGenerator instance
        embedding_generator: EmbeddingGenerator instance
        generate_profiles: Whether to generate AI profiles
        generate_embeddings: Whether to generate embeddings
        
    Returns:
        User object with profiles and embeddings
    """
    # Create user from JSON
    user = User.from_frontend_json(json_data)
    logger.info(f"Created user: {user.name}")
    
    if generate_profiles:
        # Generate dual profiles
        profile_generator.generate_dual_profiles(user)
        logger.info(f"Generated profiles for {user.name}")
    
    if generate_embeddings and user.interests_profile and user.personality_profile:
        # Generate dual embeddings
        embedding_generator.embed_user_dual(user)
        logger.info(f"Generated embeddings for {user.name}")
    
    return user


def process_frontend_json_file(filepath: str,
                             profile_generator: ProfileGenerator,
                             embedding_generator: EmbeddingGenerator) -> User:
    """
    Load and process a frontend JSON file.
    
    Args:
        filepath: Path to frontend JSON file
        profile_generator: ProfileGenerator instance
        embedding_generator: EmbeddingGenerator instance
        
    Returns:
        User object with profiles and embeddings
    """
    with open(filepath, 'r') as f:
        json_data = json.load(f)
    
    return process_frontend_json_to_user(json_data, profile_generator, embedding_generator)


def batch_process_frontend_jsons(json_files: List[str],
                                profile_generator: ProfileGenerator,
                                embedding_generator: EmbeddingGenerator) -> Population:
    """
    Process multiple frontend JSON files into a population.
    
    Args:
        json_files: List of paths to frontend JSON files
        profile_generator: ProfileGenerator instance
        embedding_generator: EmbeddingGenerator instance
        
    Returns:
        Population with processed users
    """
    population = Population(f"Frontend JSON Population ({len(json_files)} users)")
    
    for i, json_file in enumerate(json_files):
        try:
            user = process_frontend_json_file(json_file, profile_generator, embedding_generator)
            population.add_user(user)
            logger.info(f"Processed {i+1}/{len(json_files)}: {user.name}")
        except Exception as e:
            logger.error(f"Failed to process {json_file}: {e}")
    
    return population


def find_best_matches(target_user: User, 
                     population: Population, 
                     mode: str = 'combined', 
                     top_k: int = 5) -> List[Tuple[User, float, Dict[str, float]]]:
    """
    Find best matches for a user with detailed similarity breakdown.
    
    Args:
        target_user: User to find matches for
        population: Population to search in
        mode: Primary similarity mode
        top_k: Number of matches to return
        
    Returns:
        List of (User, similarity, breakdown) tuples
    """
    # Get primary matches
    matches = population.find_similar_users(target_user, top_k, mode)
    
    # Add detailed breakdown for each match
    detailed_matches = []
    for match_user, similarity in matches:
        breakdown = {}
        
        # Calculate all similarity modes
        try:
            breakdown['interests'] = target_user.calculate_similarity(match_user, 'interests')
        except ValueError:
            breakdown['interests'] = None
            
        try:
            breakdown['personality'] = target_user.calculate_similarity(match_user, 'personality')
        except ValueError:
            breakdown['personality'] = None
            
        try:
            breakdown['combined'] = target_user.calculate_similarity(match_user, 'combined')
        except ValueError:
            breakdown['combined'] = None
        
        detailed_matches.append((match_user, similarity, breakdown))
    
    return detailed_matches


def compare_similarity_modes(user1: User, user2: User) -> Dict[str, float]:
    """
    Compare two users across all similarity modes.
    
    Args:
        user1: First user
        user2: Second user
        
    Returns:
        Dictionary with similarity scores for each mode
    """
    similarities = {}
    
    modes = ['interests', 'personality', 'combined']
    if user1.embedding is not None and user2.embedding is not None:
        modes.append('legacy')
    
    for mode in modes:
        try:
            similarities[mode] = user1.calculate_similarity(user2, mode)
        except ValueError:
            similarities[mode] = None
    
    return similarities


def export_population_embeddings(population: Population, 
                                output_dir: str = "exports",
                                export_modes: List[str] = None) -> Dict[str, str]:
    """
    Export population embeddings in different formats.
    
    Args:
        population: Population to export
        output_dir: Directory to save exports
        export_modes: List of modes to export ('legacy', 'interests', 'personality', 'dual')
        
    Returns:
        Dictionary mapping mode to exported filepath
    """
    if export_modes is None:
        export_modes = ['legacy', 'interests', 'personality', 'dual']
    
    Path(output_dir).mkdir(exist_ok=True)
    exported_files = {}
    
    for mode in export_modes:
        try:
            filepath = f"{output_dir}/embeddings_{mode}.json"
            population.export_embeddings_only(filepath, mode)
            exported_files[mode] = filepath
            logger.info(f"Exported {mode} embeddings to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export {mode} embeddings: {e}")
    
    return exported_files


def analyze_population_embeddings(population: Population) -> Dict[str, Any]:
    """
    Analyze embedding distribution and quality in a population.
    
    Args:
        population: Population to analyze
        
    Returns:
        Analysis results
    """
    analysis = {
        'population_size': len(population),
        'embedding_coverage': {},
        'embedding_dimensions': {},
        'similarity_statistics': {}
    }
    
    # Check embedding coverage
    modes = ['legacy', 'interests', 'personality', 'dual']
    for mode in modes:
        users_with_embeddings = population.get_users_with_embeddings(mode)
        analysis['embedding_coverage'][mode] = {
            'count': len(users_with_embeddings),
            'percentage': len(users_with_embeddings) / len(population) * 100 if population else 0
        }
    
    # Get embedding dimensions
    for mode in ['legacy', 'interests', 'personality']:
        try:
            matrix = population.get_embedding_matrix(mode)
            analysis['embedding_dimensions'][mode] = matrix.shape[1]
        except ValueError:
            analysis['embedding_dimensions'][mode] = None
    
    # Calculate similarity statistics for each mode
    for mode in ['interests', 'personality', 'combined']:
        users_with_embeddings = population.get_users_with_embeddings(mode)
        if len(users_with_embeddings) >= 2:
            similarities = []
            
            # Sample pairwise similarities (limit to avoid O(n²) for large populations)
            sample_size = min(50, len(users_with_embeddings))
            sample_users = np.random.choice(users_with_embeddings, sample_size, replace=False)
            
            for i, user1 in enumerate(sample_users):
                for user2 in sample_users[i+1:]:
                    try:
                        sim = user1.calculate_similarity(user2, mode)
                        similarities.append(sim)
                    except ValueError:
                        continue
            
            if similarities:
                analysis['similarity_statistics'][mode] = {
                    'mean': np.mean(similarities),
                    'std': np.std(similarities),
                    'min': np.min(similarities),
                    'max': np.max(similarities),
                    'median': np.median(similarities),
                    'samples': len(similarities)
                }
    
    return analysis


def validate_dual_vector_user(user: User) -> Dict[str, bool]:
    """
    Validate that a user has all required components for dual-vector system.
    
    Args:
        user: User to validate
        
    Returns:
        Dictionary of validation results
    """
    validation = {
        'has_frontend_data': bool(user.metadata.get('frontend_data')),
        'has_interests_profile': user.interests_profile is not None,
        'has_personality_profile': user.personality_profile is not None,
        'has_interests_embedding': user.interests_embedding is not None,
        'has_personality_embedding': user.personality_embedding is not None,
        'personality_has_all_traits': False,
        'embeddings_correct_dimensions': False
    }
    
    # Check personality traits
    if user.personality_profile:
        expected_traits = {'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'}
        actual_traits = set(user.personality_profile.keys())
        validation['personality_has_all_traits'] = expected_traits.issubset(actual_traits)
    
    # Check embedding dimensions
    if user.interests_embedding is not None and user.personality_embedding is not None:
        interests_correct = len(user.interests_embedding) == 3072
        personality_correct = len(user.personality_embedding) == 3840
        validation['embeddings_correct_dimensions'] = interests_correct and personality_correct
        validation['interests_embedding_dims'] = len(user.interests_embedding)
        validation['personality_embedding_dims'] = len(user.personality_embedding)
    
    return validation


def create_compatibility_report(user1: User, user2: User) -> Dict[str, Any]:
    """
    Generate a detailed compatibility report between two users.
    
    Args:
        user1: First user
        user2: Second user
        
    Returns:
        Compatibility report
    """
    report = {
        'users': [user1.name, user2.name],
        'similarities': compare_similarity_modes(user1, user2),
        'shared_interests': {},
        'personality_alignment': {},
        'overall_compatibility': None
    }
    
    # Find shared interests from frontend data
    if user1.metadata.get('frontend_data') and user2.metadata.get('frontend_data'):
        user1_data = user1.metadata['frontend_data']
        user2_data = user2.metadata['frontend_data']
        
        # Books
        books1 = set(user1_data.get('books', {}).get('favoriteBooks', []))
        books2 = set(user2_data.get('books', {}).get('favoriteBooks', []))
        report['shared_interests']['books'] = list(books1.intersection(books2))
        
        # Movies  
        movies1 = set(user1_data.get('movies', {}).get('favoriteMovies', []))
        movies2 = set(user2_data.get('movies', {}).get('favoriteMovies', []))
        report['shared_interests']['movies'] = list(movies1.intersection(movies2))
        
        # Music
        music1 = set(user1_data.get('music', {}).get('musicArtists', []))
        music2 = set(user2_data.get('music', {}).get('musicArtists', []))
        report['shared_interests']['music'] = list(music1.intersection(music2))
    
    # Calculate overall compatibility
    valid_similarities = [s for s in report['similarities'].values() if s is not None]
    if valid_similarities:
        report['overall_compatibility'] = np.mean(valid_similarities)
    
    return report