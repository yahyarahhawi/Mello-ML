#!/usr/bin/env python3
"""
Fresh Visualizer class for 2D PCA and UMAP plotting.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Optional, Tuple, Dict, Any
import logging

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from user import User
from population import Population


class Visualizer:
    """
    Creates 2D visualizations of user embeddings using PCA and UMAP.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.logger = logging.getLogger(__name__)
        
        if not PCA_AVAILABLE:
            self.logger.warning("sklearn not available. Install with: pip install scikit-learn")
        
        if not UMAP_AVAILABLE:
            self.logger.warning("umap-learn not available. Install with: pip install umap-learn")
    
    def plot_population_pca(self, population: Population, mode: str = 'combined', 
                           highlight_special: bool = True, figsize: Tuple[int, int] = (12, 8),
                           save_path: Optional[str] = None) -> go.Figure:
        """
        Create 2D PCA plot of population embeddings using Plotly.
        
        Args:
            population: Population to visualize
            mode: Embedding mode ('combined', 'interests', or trait name)
            highlight_special: Whether to highlight special users
            figsize: Figure size (for compatibility, not used in plotly)
            save_path: Path to save plot (optional)
            
        Returns:
            plotly Figure
        """
        if not PCA_AVAILABLE:
            raise ImportError("sklearn is required for PCA. Install with: pip install scikit-learn")
        
        # Get embedding matrix
        embedding_matrix, users = population.get_embedding_matrix(mode)
        
        if len(embedding_matrix) == 0:
            raise ValueError(f"No users with {mode} embeddings found")
        
        if len(embedding_matrix) < 3:
            raise ValueError(f"Need at least 3 users for PCA, got {len(embedding_matrix)}")
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embedding_matrix)
        
        # Apply PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_scaled)
        
        # Create plotly figure
        fig = go.Figure()
        
        # Separate special and regular users
        special_indices = []
        regular_indices = []
        
        for i, user in enumerate(users):
            if user.special and highlight_special:
                special_indices.append(i)
            else:
                regular_indices.append(i)
        
        # Plot regular users (dots; hover shows name only)
        if regular_indices:
            regular_coords = embeddings_2d[regular_indices]
            regular_names = [users[i].name for i in regular_indices]

            fig.add_trace(go.Scatter(
                x=regular_coords[:, 0],
                y=regular_coords[:, 1],
                mode='markers',
                marker=dict(size=8, color='lightblue', opacity=0.7, line=dict(width=1, color='darkblue')),
                name=f'Users ({len(regular_indices)})',
                hovertext=regular_names,                        # ⬅️ drive tooltip from hovertext
                hovertemplate='<b>%{hovertext}</b><extra></extra>'
            ))
        
        # Plot special users (star markers; hover shows name only)
        if special_indices:
            special_coords = embeddings_2d[special_indices]
            special_names = [users[i].name for i in special_indices]

            fig.add_trace(go.Scatter(
                x=special_coords[:, 0],
                y=special_coords[:, 1],
                mode='markers',
                marker=dict(size=15, color='red', opacity=0.9, symbol='star', line=dict(width=2, color='darkred')),
                name=f'Special Users ({len(special_indices)})',
                hovertext=special_names,                        # ⬅️ only names
                hovertemplate='<b>%{hovertext}</b><extra></extra>'
            ))
        
        # Update layout
        total_variance = pca.explained_variance_ratio_[:2].sum()
        fig.update_layout(
            title=f'PCA Visualization - {mode.title()} Embeddings<br>{population.name} ({len(users)} users)',
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
            hovermode='closest',
            showlegend=True,
            width=figsize[0] * 80,
            height=figsize[1] * 80,
            annotations=[
                dict(
                    text=f'Total variance explained: {total_variance:.1%}<br>Similarity metric: Euclidean Distance',
                    xref="paper", yref="paper",
                    x=0.02, y=0.98, xanchor='left', yanchor='top',
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.5)",
                    borderwidth=1
                )
            ]
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Saved PCA plot to {save_path}")
        
        return fig
    
    def plot_population_umap(self, population: Population, mode: str = 'combined',
                            highlight_special: bool = True, figsize: Tuple[int, int] = (12, 8),
                            n_neighbors: int = 15, min_dist: float = 0.1, metric: str = 'euclidean',
                            save_path: Optional[str] = None) -> go.Figure:
        """
        Create 2D UMAP plot of population embeddings using Plotly.
        
        Args:
            population: Population to visualize
            mode: Embedding mode ('combined', 'interests', or trait name)
            highlight_special: Whether to highlight special users
            figsize: Figure size (for compatibility, not used in plotly)
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            metric: UMAP metric parameter (default: euclidean)
            save_path: Path to save plot (optional)
            
        Returns:
            plotly Figure
        """
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn is required for UMAP. Install with: pip install umap-learn")
        
        # Get embedding matrix
        embedding_matrix, users = population.get_embedding_matrix(mode)
        
        if len(embedding_matrix) == 0:
            raise ValueError(f"No users with {mode} embeddings found")
        
        if len(embedding_matrix) < 10:
            raise ValueError(f"Need at least 10 users for UMAP, got {len(embedding_matrix)}")
        
        # Apply UMAP
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, 
                           metric=metric, random_state=42)
        embeddings_2d = reducer.fit_transform(embedding_matrix)
        
        # Create plotly figure
        fig = go.Figure()
        
        # Separate special and regular users
        special_indices = []
        regular_indices = []
        
        for i, user in enumerate(users):
            if user.special and highlight_special:
                special_indices.append(i)
            else:
                regular_indices.append(i)
        
        # Plot regular users (dots; hover shows name only)
        if regular_indices:
            regular_coords = embeddings_2d[regular_indices]
            regular_names = [users[i].name for i in regular_indices]

            fig.add_trace(go.Scatter(
                x=regular_coords[:, 0],
                y=regular_coords[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color='lightblue',
                    opacity=0.7,
                    line=dict(width=1, color='darkblue')
                ),
                name=f'Users ({len(regular_indices)})',
                text=regular_names,
                hovertemplate='<b>%{text}</b><extra></extra>'
            ))
        
        # Plot special users (star markers; hover shows name only)
        if special_indices:
            special_coords = embeddings_2d[special_indices]
            special_names = [users[i].name for i in special_indices]

            fig.add_trace(go.Scatter(
                x=special_coords[:, 0],
                y=special_coords[:, 1],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    opacity=0.9,
                    symbol='star',
                    line=dict(width=2, color='darkred')
                ),
                name=f'Special Users ({len(special_indices)})',
                text=special_names,
                hovertemplate='<b>%{text}</b><extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'UMAP Visualization - {mode.title()} Embeddings<br>{population.name} ({len(users)} users)',
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            hovermode='closest',
            showlegend=True,
            width=figsize[0] * 80,
            height=figsize[1] * 80,
            annotations=[
                dict(
                    text=f'n_neighbors={n_neighbors}, min_dist={min_dist}<br>metric={metric}, similarity: Euclidean Distance',
                    xref="paper", yref="paper",
                    x=0.02, y=0.98, xanchor='left', yanchor='top',
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.5)",
                    borderwidth=1
                )
            ]
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Saved UMAP plot to {save_path}")
        
        return fig
    
    def plot_similarity_comparison(self, target_user: User, similar_users: List[Tuple[User, float]], 
                                  mode: str = 'combined', figsize: Tuple[int, int] = (10, 6),
                                  save_path: Optional[str] = None) -> go.Figure:
        """
        Plot similarity scores for similar users using Plotly.
        
        Args:
            target_user: Target user
            similar_users: List of (user, similarity_score) tuples
            mode: Embedding mode used
            figsize: Figure size (for compatibility, not used in plotly)
            save_path: Path to save plot (optional)
            
        Returns:
            plotly Figure
        """
        if not similar_users:
            raise ValueError("No similar users provided")
        
        # Extract data
        users, scores = zip(*similar_users)
        user_names = [user.name for user in users]
        
        # Create colors - highlight most similar user
        colors = ['orange' if i == 0 else 'lightblue' for i in range(len(users))]
        
        # Create bar plot
        fig = go.Figure(data=[
            go.Bar(
                x=user_names,
                y=scores,
                marker_color=colors,
                marker_line_color='navy',
                marker_line_width=1,
                text=[f'{score:.3f}' for score in scores],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Similarity: %{y:.3f}<extra></extra>'
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=f'Most Similar Users to {target_user.name}<br>({mode.title()} Embeddings, Euclidean Distance)',
            xaxis_title='Users',
            yaxis_title='Similarity Score',
            yaxis=dict(range=[0, 1], gridcolor='lightgray', gridwidth=1),
            width=figsize[0] * 80,
            height=figsize[1] * 80,
            showlegend=False
        )
        
        # Rotate x-axis labels if there are many users
        if len(user_names) > 5:
            fig.update_xaxes(tickangle=45)
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Saved similarity plot to {save_path}")
        
        return fig
    
    def plot_trait_comparison(self, users: List[User], figsize: Tuple[int, int] = (14, 8),
                             save_path: Optional[str] = None) -> go.Figure:
        """
        Plot trait embedding similarities across multiple users.
        
        Args:
            users: List of users to compare
            figsize: Figure size
            save_path: Path to save plot (optional)
            
        Returns:
            matplotlib Figure
        """
        traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        
        # Calculate pairwise similarities for each trait
        trait_similarities = {}
        
        for trait in traits:
            similarities = []
            for i, user1 in enumerate(users):
                for j, user2 in enumerate(users):
                    if i < j:  # Only upper triangle
                        try:
                            sim = user1.calculate_similarity(user2, trait)
                            similarities.append(sim)
                        except:
                            similarities.append(0.0)
            
            trait_similarities[trait] = similarities
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Box plot of trait similarities
        data = [trait_similarities[trait] for trait in traits]
        box_plot = ax.boxplot(data, labels=traits, patch_artist=True)
        
        # Color the boxes
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightpink']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Formatting
        ax.set_ylabel('Similarity Score')
        ax.set_title(f'Trait Embedding Similarities\\n({len(users)} users, {len(similarities)} pairwise comparisons per trait)')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean lines
        for i, trait in enumerate(traits):
            mean_sim = np.mean(trait_similarities[trait])
            ax.hlines(mean_sim, i + 0.8, i + 1.2, colors='red', linestyles='dashed', alpha=0.8)
            ax.text(i + 1, mean_sim + 0.02, f'{mean_sim:.3f}', ha='center', 
                   va='bottom', fontsize=10, color='red', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved trait comparison plot to {save_path}")
        
        return fig
    
    def create_embedding_summary(self, population: Population) -> Dict[str, Any]:
        """
        Create summary statistics of embeddings in the population.
        
        Args:
            population: Population to analyze
            
        Returns:
            Dictionary with embedding statistics
        """
        summary = {
            'population_name': population.name,
            'total_users': len(population),
            'users_with_embeddings': len(population.get_users_with_embeddings()),
            'embedding_modes': {}
        }
        
        # Check each embedding mode
        modes = ['combined', 'interests', 'Openness', 'Conscientiousness', 
                'Extraversion', 'Agreeableness', 'Neuroticism']
        
        for mode in modes:
            try:
                embedding_matrix, users = population.get_embedding_matrix(mode)
                if len(embedding_matrix) > 0:
                    summary['embedding_modes'][mode] = {
                        'users_count': len(users),
                        'dimensions': embedding_matrix.shape[1],
                        'mean_norm': float(np.mean(np.linalg.norm(embedding_matrix, axis=1))),
                        'std_norm': float(np.std(np.linalg.norm(embedding_matrix, axis=1)))
                    }
                else:
                    summary['embedding_modes'][mode] = {'users_count': 0}
            except Exception as e:
                summary['embedding_modes'][mode] = {'error': str(e)}
        
        return summary
    
    def plot_interactive_population(self, population: Population, modes: List[str] = None, 
                                   figsize: Tuple[int, int] = (15, 10)):
        """
        Interactive population visualization (disabled - use Plotly methods instead).
        
        Args:
            population: Population to visualize
            modes: List of embedding modes to include
            figsize: Figure size
            
        Returns:
            None (method disabled)
        """
        raise NotImplementedError("Interactive population plotting is disabled. Use plot_population_pca() or plot_population_umap() with Plotly instead.")
    
    def plot_similarity_heatmap(self, users: List[User], mode: str = 'combined', 
                               figsize: Tuple[int, int] = (12, 10)) -> go.Figure:
        """
        Create an interactive similarity heatmap between users using Plotly.
        
        Args:
            users: List of users to compare
            mode: Embedding mode to use
            figsize: Figure size (for compatibility, not used in plotly)
            
        Returns:
            plotly Figure with heatmap
        """
        if len(users) < 2:
            raise ValueError("Need at least 2 users for similarity heatmap")
        
        # Calculate similarity matrix
        n_users = len(users)
        similarity_matrix = np.zeros((n_users, n_users))
        
        for i in range(n_users):
            for j in range(n_users):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity_matrix[i, j] = users[i].calculate_similarity(users[j], mode)
        
        # Create user labels
        user_labels = [f"{user.name[:15]}" for user in users]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=user_labels,
            y=user_labels,
            colorscale='RdYlBu_r',
            zmin=0,
            zmax=1,
            text=[[f'{similarity_matrix[i, j]:.2f}' for j in range(n_users)] for i in range(n_users)],
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Similarity: %{z:.3f}<extra></extra>',
            colorbar=dict(
                title="Euclidean Distance<br>Similarity",
                titleside="right"
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=f'User Similarity Heatmap - {mode.title()} Embeddings<br>Euclidean Distance Similarity (0=different, 1=identical)',
            xaxis_title='Users',
            yaxis_title='Users',
            width=figsize[0] * 80,
            height=figsize[1] * 80,
            xaxis=dict(tickangle=45),
            yaxis=dict(autorange='reversed')  # Reverse y-axis to match matplotlib convention
        )
        
        return fig