#!/usr/bin/env python3
"""
Visualizer class for dimensionality reduction and interactive plotting.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from population import Population
from user import User


class Visualizer:
    """
    Handles dimensionality reduction and visualization of user embeddings
    using PCA, t-SNE, and UMAP.
    """
    
    def __init__(self, population: Population = None):
        """
        Initialize the visualizer.
        
        Args:
            population: Population to visualize
        """
        self.population = population
        self.logger = logging.getLogger(__name__)
        
        # Store fitted models for transforming new data
        self.pca_2d = None
        self.pca_3d = None
        self.tsne_2d = None
        self.tsne_3d = None
        self.umap_2d = None
        self.umap_3d = None
        self.scaler = None
        
        # Store transformed coordinates
        self.coordinates = {}
        
        if not UMAP_AVAILABLE:
            self.logger.warning("UMAP not available. Install with: pip install umap-learn")
    
    def set_population(self, population: Population):
        """Set or update the population to visualize."""
        self.population = population
        self.coordinates = {}  # Clear cached coordinates
    
    def _prepare_data(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Prepare embedding data for visualization.
        
        Returns:
            Tuple of (embeddings_matrix, user_names, taste_profiles)
        """
        if not self.population:
            raise ValueError("No population set for visualization")
        
        users_with_embeddings = self.population.get_users_with_embeddings()
        if not users_with_embeddings:
            raise ValueError("No users with embeddings found")
        
        embeddings = np.array([user.embedding for user in users_with_embeddings])
        names = [user.name for user in users_with_embeddings]
        profiles = [user.taste_profile or "No profile" for user in users_with_embeddings]
        
        return embeddings, names, profiles
    
    def _standardize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Standardize embeddings for better dimensionality reduction."""
        if self.scaler is None:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(embeddings)
        else:
            return self.scaler.transform(embeddings)
    
    def fit_pca(self, n_components_2d: int = 2, n_components_3d: int = 3) -> Dict[str, np.ndarray]:
        """
        Fit PCA models and return transformed coordinates.
        
        Args:
            n_components_2d: Number of components for 2D PCA
            n_components_3d: Number of components for 3D PCA
            
        Returns:
            Dictionary with 'pca_2d' and 'pca_3d' coordinates
        """
        embeddings, names, profiles = self._prepare_data()
        embeddings_scaled = self._standardize_embeddings(embeddings)
        
        # Fit 2D PCA
        self.pca_2d = PCA(n_components=n_components_2d, random_state=42)
        coords_2d = self.pca_2d.fit_transform(embeddings_scaled)
        
        # Fit 3D PCA
        self.pca_3d = PCA(n_components=n_components_3d, random_state=42)
        coords_3d = self.pca_3d.fit_transform(embeddings_scaled)
        
        self.coordinates['pca_2d'] = coords_2d
        self.coordinates['pca_3d'] = coords_3d
        
        self.logger.info(f"PCA fitted. 2D explained variance: {self.pca_2d.explained_variance_ratio_.sum():.3f}")
        self.logger.info(f"PCA fitted. 3D explained variance: {self.pca_3d.explained_variance_ratio_.sum():.3f}")
        
        return {'pca_2d': coords_2d, 'pca_3d': coords_3d}
    
    def fit_tsne(self, n_components_2d: int = 2, n_components_3d: int = 3, 
                 perplexity: float = 30.0, n_iter: int = 1000) -> Dict[str, np.ndarray]:
        """
        Fit t-SNE models and return transformed coordinates.
        
        Args:
            n_components_2d: Number of components for 2D t-SNE
            n_components_3d: Number of components for 3D t-SNE
            perplexity: t-SNE perplexity parameter
            n_iter: Number of iterations
            
        Returns:
            Dictionary with 'tsne_2d' and 'tsne_3d' coordinates
        """
        embeddings, names, profiles = self._prepare_data()
        embeddings_scaled = self._standardize_embeddings(embeddings)
        
        # Fit 2D t-SNE
        self.tsne_2d = TSNE(n_components=n_components_2d, perplexity=perplexity, 
                           n_iter=n_iter, random_state=42, init='pca')
        coords_2d = self.tsne_2d.fit_transform(embeddings_scaled)
        
        # Fit 3D t-SNE
        self.tsne_3d = TSNE(n_components=n_components_3d, perplexity=perplexity, 
                           n_iter=n_iter, random_state=42, init='pca')
        coords_3d = self.tsne_3d.fit_transform(embeddings_scaled)
        
        self.coordinates['tsne_2d'] = coords_2d
        self.coordinates['tsne_3d'] = coords_3d
        
        self.logger.info(f"t-SNE fitted with perplexity={perplexity}, n_iter={n_iter}")
        
        return {'tsne_2d': coords_2d, 'tsne_3d': coords_3d}
    
    def fit_umap(self, n_components_2d: int = 2, n_components_3d: int = 3,
                 n_neighbors: int = 15, min_dist: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Fit UMAP models and return transformed coordinates.
        
        Args:
            n_components_2d: Number of components for 2D UMAP
            n_components_3d: Number of components for 3D UMAP
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            
        Returns:
            Dictionary with 'umap_2d' and 'umap_3d' coordinates
        """
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        
        embeddings, names, profiles = self._prepare_data()
        embeddings_scaled = self._standardize_embeddings(embeddings)
        
        # Fit 2D UMAP
        self.umap_2d = umap.UMAP(n_components=n_components_2d, n_neighbors=n_neighbors,
                                min_dist=min_dist, random_state=42)
        coords_2d = self.umap_2d.fit_transform(embeddings_scaled)
        
        # Fit 3D UMAP
        self.umap_3d = umap.UMAP(n_components=n_components_3d, n_neighbors=n_neighbors,
                                min_dist=min_dist, random_state=42)
        coords_3d = self.umap_3d.fit_transform(embeddings_scaled)
        
        self.coordinates['umap_2d'] = coords_2d
        self.coordinates['umap_3d'] = coords_3d
        
        self.logger.info(f"UMAP fitted with n_neighbors={n_neighbors}, min_dist={min_dist}")
        
        return {'umap_2d': coords_2d, 'umap_3d': coords_3d}
    
    def transform_new_embedding(self, embedding: np.ndarray, method: str) -> np.ndarray:
        """
        Transform a new embedding using fitted models.
        
        Args:
            embedding: New embedding to transform
            method: Method to use ('pca_2d', 'pca_3d', 'umap_2d', 'umap_3d')
            
        Returns:
            Transformed coordinates
        """
        if self.scaler is None:
            raise ValueError("Models not fitted yet. Call fit_* methods first.")
        
        # Standardize the new embedding
        embedding_scaled = self.scaler.transform(embedding.reshape(1, -1))
        
        if method == 'pca_2d' and self.pca_2d is not None:
            return self.pca_2d.transform(embedding_scaled)[0]
        elif method == 'pca_3d' and self.pca_3d is not None:
            return self.pca_3d.transform(embedding_scaled)[0]
        elif method == 'umap_2d' and self.umap_2d is not None:
            return self.umap_2d.transform(embedding_scaled)[0]
        elif method == 'umap_3d' and self.umap_3d is not None:
            return self.umap_3d.transform(embedding_scaled)[0]
        else:
            raise ValueError(f"Method '{method}' not available or not fitted")
    
    def plot_2d(self, method: str = 'umap_2d', highlight_users: List[str] = None,
                title: str = None, save_path: str = None) -> go.Figure:
        """
        Create 2D scatter plot.
        
        Args:
            method: Dimensionality reduction method ('pca_2d', 'tsne_2d', 'umap_2d')
            highlight_users: List of user names to highlight
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Plotly figure
        """
        if method not in self.coordinates:
            raise ValueError(f"Method '{method}' not fitted yet")
        
        coords = self.coordinates[method]
        embeddings, names, profiles = self._prepare_data()
        
        # Create colors for highlighting
        colors = ['blue'] * len(names)
        if highlight_users:
            for i, name in enumerate(names):
                if name in highlight_users:
                    colors[i] = 'red'
        
        fig = go.Figure()
        
        # Add regular points
        regular_mask = [c == 'blue' for c in colors]
        if any(regular_mask):
            fig.add_trace(go.Scatter(
                x=coords[regular_mask, 0],
                y=coords[regular_mask, 1],
                mode='markers',
                marker=dict(color='blue', size=8, opacity=0.7),
                text=[names[i] for i in range(len(names)) if regular_mask[i]],
                hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>',
                name='Users'
            ))
        
        # Add highlighted points
        highlight_mask = [c == 'red' for c in colors]
        if any(highlight_mask):
            fig.add_trace(go.Scatter(
                x=coords[highlight_mask, 0],
                y=coords[highlight_mask, 1],
                mode='markers',
                marker=dict(color='red', size=12, opacity=0.9),
                text=[names[i] for i in range(len(names)) if highlight_mask[i]],
                hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>',
                name='Highlighted'
            ))
        
        fig.update_layout(
            title=title or f'{method.upper()} Visualization',
            xaxis_title=f'{method} Component 1',
            yaxis_title=f'{method} Component 2',
            width=800,
            height=600,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_3d(self, method: str = 'umap_3d', highlight_users: List[str] = None,
                title: str = None, save_path: str = None) -> go.Figure:
        """
        Create 3D scatter plot.
        
        Args:
            method: Dimensionality reduction method ('pca_3d', 'tsne_3d', 'umap_3d')
            highlight_users: List of user names to highlight
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Plotly figure
        """
        if method not in self.coordinates:
            raise ValueError(f"Method '{method}' not fitted yet")
        
        coords = self.coordinates[method]
        embeddings, names, profiles = self._prepare_data()
        
        # Create colors for highlighting
        colors = ['blue'] * len(names)
        if highlight_users:
            for i, name in enumerate(names):
                if name in highlight_users:
                    colors[i] = 'red'
        
        fig = go.Figure()
        
        # Add regular points
        regular_mask = [c == 'blue' for c in colors]
        if any(regular_mask):
            fig.add_trace(go.Scatter3d(
                x=coords[regular_mask, 0],
                y=coords[regular_mask, 1],
                z=coords[regular_mask, 2],
                mode='markers',
                marker=dict(color='blue', size=5, opacity=0.7),
                text=[names[i] for i in range(len(names)) if regular_mask[i]],
                customdata=[profiles[i] for i in range(len(profiles)) if regular_mask[i]],
                hovertemplate='<b>%{text}</b><br>%{customdata}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
                name='Users'
            ))
        
        # Add highlighted points
        highlight_mask = [c == 'red' for c in colors]
        if any(highlight_mask):
            fig.add_trace(go.Scatter3d(
                x=coords[highlight_mask, 0],
                y=coords[highlight_mask, 1],
                z=coords[highlight_mask, 2],
                mode='markers',
                marker=dict(color='red', size=8, opacity=0.9),
                text=[names[i] for i in range(len(names)) if highlight_mask[i]],
                customdata=[profiles[i] for i in range(len(profiles)) if highlight_mask[i]],
                hovertemplate='<b>%{text}</b><br>%{customdata}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
                name='Highlighted'
            ))
        
        fig.update_layout(
            title=title or f'{method.upper()} 3D Visualization',
            scene=dict(
                xaxis_title=f'{method} Component 1',
                yaxis_title=f'{method} Component 2',
                zaxis_title=f'{method} Component 3'
            ),
            width=900,
            height=700,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_comparison(self, methods: List[str] = ['pca_2d', 'tsne_2d', 'umap_2d'],
                       save_path: str = None) -> go.Figure:
        """
        Create comparison plot of different dimensionality reduction methods.
        
        Args:
            methods: List of methods to compare
            save_path: Path to save the plot
            
        Returns:
            Plotly figure with subplots
        """
        n_methods = len(methods)
        fig = make_subplots(
            rows=1, cols=n_methods,
            subplot_titles=[method.upper() for method in methods],
            specs=[[{"type": "scatter"}] * n_methods]
        )
        
        embeddings, names, profiles = self._prepare_data()
        
        for i, method in enumerate(methods):
            if method not in self.coordinates:
                continue
            
            coords = self.coordinates[method]
            
            fig.add_trace(
                go.Scatter(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    mode='markers',
                    marker=dict(size=6, opacity=0.7),
                    text=names,
                    hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>',
                    name=method.upper()
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title='Dimensionality Reduction Comparison',
            height=400,
            width=300 * n_methods,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Saved comparison plot to {save_path}")
        
        return fig
    
    def find_closest_in_space(self, target_coords: np.ndarray, method: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Find closest users to target coordinates in the reduced space.
        
        Args:
            target_coords: Target coordinates in the reduced space
            method: Method space to search in
            k: Number of closest users to return
            
        Returns:
            List of (username, distance) tuples
        """
        if method not in self.coordinates:
            raise ValueError(f"Method '{method}' not fitted yet")
        
        coords = self.coordinates[method]
        embeddings, names, profiles = self._prepare_data()
        
        # Calculate distances
        distances = np.linalg.norm(coords - target_coords, axis=1)
        
        # Get k closest
        closest_indices = np.argsort(distances)[:k]
        
        return [(names[i], distances[i]) for i in closest_indices]
    
    def get_method_statistics(self) -> Dict[str, Any]:
        """Get statistics about fitted methods."""
        stats = {}
        
        if self.pca_2d is not None:
            stats['pca_2d_explained_variance'] = self.pca_2d.explained_variance_ratio_.sum()
        
        if self.pca_3d is not None:
            stats['pca_3d_explained_variance'] = self.pca_3d.explained_variance_ratio_.sum()
        
        stats['fitted_methods'] = list(self.coordinates.keys())
        
        return stats
    
    def __str__(self) -> str:
        """String representation."""
        n_users = len(self.population.users) if self.population else 0
        fitted_methods = list(self.coordinates.keys())
        return f"Visualizer(users={n_users}, methods={fitted_methods})"