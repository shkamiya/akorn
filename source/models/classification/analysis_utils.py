"""
Connectivity Analysis Utilities for MyAKOrN Models

This module provides classes and functions for analyzing the connectivity 
patterns and dynamics of learned AKOrN (Artificial Kuramoto Oscillator Network) models.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

class AKOrNStaticAnalyzer:
    """
    A comprehensive analyzer for MyAKOrN model connectivity patterns.
    
    This class provides methods to inspect and analyze the learned connectivity
    matrices, omega parameters, and their geometric properties.
    """
    
    def __init__(self, model, layer_idx: int, device: str = 'cpu'):
        """
        Initialize the connectivity analyzer for a specific layer.
        
        Args:
            model: Trained MyAKOrN model
            layer_idx: Layer index to analyze
            device: Device to run analysis on ('cpu' or 'cuda')
        """
        self.model = model
        self.layer_idx = layer_idx
        self.device = device
        self.model.eval()
        
        # Extract basic model parameters
        self.n = 2  # oscillator dimension (assumed to be 2)
        self.num_layers = len(model.layers)
        
        # Validate layer index
        if layer_idx >= self.num_layers or layer_idx < 0:
            raise ValueError(f"Layer index {layer_idx} is out of range [0, {self.num_layers-1}]")
        
        # Storage for extracted parameters (single layer)
        self.omega_param = None
        self.connectivity_weight = None
        self.connectivity_blocks = None
        self.decomposition_results = None
        
        # Analysis results
        self.clustering_results = None
        self.dimensionality_reduction_results = None
        
    def extract_omega_parameters(self) -> Optional[np.ndarray]:
        """Extract omega parameters from the specified layer."""
        if self.omega_param is not None:
            return self.omega_param
            
        layer = self.model.layers[self.layer_idx]
        if hasattr(layer[2], 'omg') and hasattr(layer[2].omg, 'omg_param'):
            self.omega_param = layer[2].omg.omg_param.detach().cpu().numpy()
            print(f"Layer {self.layer_idx}: omega shape = {self.omega_param.shape}")
            print(f"  Omega magnitude: {np.linalg.norm(self.omega_param):.4f}")
        
        return self.omega_param
    
    def extract_connectivity_weights(self) -> Optional[Dict]:
        """Extract connectivity weight matrices from the specified layer."""
        if self.connectivity_weight is not None:
            return self.connectivity_weight
            
        layer = self.model.layers[self.layer_idx]
        if hasattr(layer[2], 'connectivity'):
            weight = layer[2].connectivity.weight.detach().cpu().numpy()
            bias = layer[2].connectivity.bias.detach().cpu().numpy() if layer[2].connectivity.bias is not None else None
            
            self.connectivity_weight = {
                'weight': weight,
                'bias': bias,
                'shape': weight.shape,
                'layer_idx': self.layer_idx
            }
            
            print(f"Layer {self.layer_idx}: Connectivity weight shape = {weight.shape}")
            print(f"  Weight statistics: mean={weight.mean():.4f}, std={weight.std():.4f}")
            print(f"  Weight range: [{weight.min():.4f}, {weight.max():.4f}]")
        
        return self.connectivity_weight
    
    def extract_connectivity_blocks(self) -> np.ndarray:
        """
        Extract 2x2 connectivity blocks from the specified layer.
        
        Returns:
            Array of shape (num_blocks, 2, 2) containing all 2x2 connectivity blocks
        """
        # Return cached blocks if available
        if self.connectivity_blocks is not None:
            return self.connectivity_blocks
            
        # Ensure connectivity weights are extracted
        if self.connectivity_weight is None:
            self.extract_connectivity_weights()
            
        if self.connectivity_weight is None:
            raise ValueError(f"No connectivity weights found for layer {self.layer_idx}")
            
        J = self.connectivity_weight['weight']
        C_out, C_in, H, W = J.shape
        
        # Reshape to extract 2x2 blocks
        blocks = (
            J.reshape(C_out//self.n, self.n, C_in//self.n, self.n, H, W)
             .transpose(0, 2, 4, 5, 1, 3)  # (64,64,9,9,2,2)
             .reshape(-1, 2, 2)            # (num_blocks, 2, 2)
        )
        
        # Cache the blocks
        self.connectivity_blocks = blocks
        return blocks
    
    def decompose_symmetric_skew(self, J: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose connectivity matrices into symmetric + skew-symmetric parts.
        
        Args:
            J: Connectivity matrix/matrices of shape (2,2) or (n,2,2)
            
        Returns:
            Tuple of (p1, p2, p3, q) where:
            - p1, p2, p3 are symmetric part components
            - q is skew-symmetric part component
        """
        J = np.asarray(J)
        J_sym = (J + np.swapaxes(J, -2, -1)) / 2
        J_skew = (J - np.swapaxes(J, -2, -1)) / 2
        # J_sym = (J + np.transpose(J, axes=(-1, -2))) / 2
        # J_skew = (J - np.transpose(J, axes=(-1, -2))) / 2
        
        if J.ndim == 2 and J.shape == (2, 2):
            p1, p2, p3 = J_sym[0,0], J_sym[0,1], J_sym[1,1]
            q = J_skew[1,0]
        elif J.ndim == 3 and J.shape[1:] == (2, 2):
            p1 = J_sym[:, 0, 0]
            p2 = J_sym[:, 0, 1]
            p3 = J_sym[:, 1, 1]
            q = J_skew[:, 1, 0]
        else:
            raise ValueError("Input must be shape (2,2) or (n,2,2)")
        
        # Compute Frobenius norms of symmetric_skew
        sym_frob = np.sqrt(p1**2 + 2*p2**2 + p3**2)  # ||J_sym||_F = sqrt(p1^2 + 2*p2^2 + p3^2)
        skew_frob = np.sqrt(2) * np.abs(q)  # ||J_skew||_F = sqrt(2) * |q|
        
        return p1, p2, p3, q, sym_frob, skew_frob
    
    def decompose_rotation_symmetric(self, J: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose connectivity matrices into rotation + symmetric parts.
        
        Following Costa & Aguiar 2024 decomposition:
        J = c_R * R(alpha) + c_S * S(beta)
        
        Args:
            J: Connectivity matrix/matrices of shape (2,2) or (n,2,2)
            
        Returns:
            Tuple of (c_R, c_S, alpha, beta)
        """
        J = np.asarray(J)
        if J.ndim == 2 and J.shape == (2, 2):
            a, b = J[0, 0], J[0, 1]
            c, d = J[1, 0], J[1, 1]
        elif J.ndim == 3 and J.shape[1:] == (2, 2):
            a = J[:, 0, 0]
            b = J[:, 0, 1]
            c = J[:, 1, 0]
            d = J[:, 1, 1]
        else:
            raise ValueError("Input must be shape (2,2) or (n,2,2)")
            
        c_R = 0.5 * np.sqrt((a + d)**2 + (b - c)**2)
        c_S = 0.5 * np.sqrt((d - a)**2 + (b + c)**2)
        alpha = np.arctan2(b - c, a + d)
        beta = np.arctan2(b + c, d - a)
        
        return c_R, c_S, alpha, beta
    
    def analyze_connectivity_clustering(self, k_range: range = range(2, 11), 
                                       standardize: bool = False, random_state: int = 0) -> Dict:
        """
        Perform clustering analysis on connectivity blocks.
        
        Args:
            k_range: Range of k values to test for clustering
            standardize: Whether to standardize features before clustering
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing clustering results
        """
        # Extract connectivity blocks for the specified layer
        connectivity_blocks = self.extract_connectivity_blocks()
        
        # Flatten connectivity blocks to feature vectors
        X = connectivity_blocks.reshape(connectivity_blocks.shape[0], -1)
        
        if standardize:
            X = StandardScaler().fit_transform(X)
        
        # Find optimal k using silhouette score
        best_k, best_score = None, -1
        scores = []
        
        for k in k_range:
            km = KMeans(n_clusters=k, n_init='auto', random_state=random_state).fit(X)
            score = silhouette_score(X, km.labels_, sample_size=min(10000, X.shape[0]))
            scores.append(score)
            
            if score > best_score:
                best_k, best_score, best_model = k, score, km
        
        # Store results
        self.clustering_results = {
            'k_range': list(k_range),
            'silhouette_scores': scores,
            'best_k': best_k,
            'best_score': best_score,
            'best_model': best_model,
            'labels': best_model.labels_,
            'centers': best_model.cluster_centers_.reshape(best_k, 2, 2),
            'layer_idx': self.layer_idx
        }
        
        return self.clustering_results
    
    def perform_dimensionality_reduction(self, methods: List[str] = ['pca']) -> Dict:
        """
        Perform dimensionality reduction on connectivity blocks.
        
        Args:
            methods: List of methods to use ('pca', 'tsne', 'umap')
            
        Returns:
            Dictionary containing dimensionality reduction results
        """
        # Return cached results if available
        if self.dimensionality_reduction_results is not None:
            return self.dimensionality_reduction_results
            
        # Extract connectivity blocks for the specified layer  
        connectivity_blocks = self.extract_connectivity_blocks()
        
        X = connectivity_blocks.reshape(connectivity_blocks.shape[0], -1)
        results = {}
        
        if 'pca' in methods:
            pca = PCA(n_components=3, random_state=0)
            X_pca = pca.fit_transform(X)
            results['pca'] = {
                'embedding': X_pca,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'model': pca
            }
        
        if 'tsne' in methods:
            # Sample for t-SNE if dataset is large
            n_samples = min(20000, X.shape[0])
            rng = np.random.default_rng(0)
            idx_vis = rng.choice(X.shape[0], size=n_samples, replace=False)
            X_vis = X[idx_vis]
            
            tsne = TSNE(n_components=2, perplexity=30, init='pca', 
                       learning_rate='auto', random_state=0)
            X_tsne = tsne.fit_transform(X_vis)
            results['tsne'] = {
                'embedding': X_tsne,
                'sample_indices': idx_vis,
                'model': tsne
            }
        
        if 'umap' in methods and HAS_UMAP:
            n_samples = min(20000, X.shape[0])
            rng = np.random.default_rng(0)
            idx_vis = rng.choice(X.shape[0], size=n_samples, replace=False)
            X_vis = X[idx_vis]
            
            mapper = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                             metric="euclidean", random_state=0)
            X_umap = mapper.fit_transform(X_vis)
            results['umap'] = {
                'embedding': X_umap,
                'sample_indices': idx_vis,
                'model': mapper
            }
        
        # Cache results
        self.dimensionality_reduction_results = results
        return results
    
    def analyze_full_connectivity(self) -> Dict:
        """
        Perform comprehensive connectivity analysis for the specified layer.
        
        Returns:
            Dictionary containing all analysis results
        """
        # Extract connectivity blocks for the specified layer
        blocks = self.extract_connectivity_blocks()
        
        # Perform decompositions
        p1, p2, p3, q, sym_frob, skew_frob = self.decompose_symmetric_skew(blocks)
        c_R, c_S, alpha, beta = self.decompose_rotation_symmetric(blocks)

        # Store decomposition results
        self.decomposition_results = {
            'layer_idx': self.layer_idx,
            'blocks': blocks,
            'symmetric_skew': {'p1': p1, 'p2': p2, 'p3': p3, 'q': q, 'sym_frob': sym_frob, 'skew_frob': skew_frob},
            'rotation_symmetric': {'c_R': c_R, 'c_S': c_S, 'alpha': alpha, 'beta': beta}
        }
        
        # Perform clustering
        # clustering_results = self.analyze_connectivity_clustering()
        
        # Perform dimensionality reduction
        if self.dimensionality_reduction_results is None:
            self.perform_dimensionality_reduction(['pca'])
        
        return {
            'decomposition': self.decomposition_results,
            'clustering': self.clustering_results,
            'dimensionality_reduction': self.dimensionality_reduction_results
        }
    
    def plot_omega_distributions(self, figsize: Tuple[int, int] = (8, 4), show: bool = True) -> None:
        """Plot omega parameter distributions for the specified layer."""
        if self.omega_param is None:
            self.extract_omega_parameters()
            
        if self.omega_param is None:
            print(f"No omega parameters found for layer {self.layer_idx}")
            return
            
        plt.figure(figsize=figsize)
        omega_vals = self.omega_param[:, 0]  # Take first column
        plt.hist(omega_vals, bins=20, color='C0', alpha=0.7)
        plt.title(f'Layer {self.layer_idx} Omega Histogram')
        plt.xlabel('Omega Value')
        plt.ylabel('Count')
            
        plt.tight_layout()
        if show:
            plt.show()
    

    def plot_connectivity_statistics(self, show: bool = True) -> None:
        """Plot summary statistics of connectivity matrices."""
        # Extract connectivity blocks for the specified layer
        connectivity_blocks = self.extract_connectivity_blocks()
        X = connectivity_blocks.reshape(connectivity_blocks.shape[0], -1)    # (many, 4)
        df_X = pd.DataFrame({
            'J11': X[:, 0],
            'J12': X[:, 1],
            'J21': X[:, 2],
            'J22': X[:, 3]
        })

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left subplot: violin plot of 2x2 block elements
        sns.violinplot(data=df_X, ax=axes[0], inner='quartile')
        axes[0].set_title(f'Layer {self.layer_idx}: Distribution of 2x2 Block Elements')
        axes[0].set_ylabel('Value')
        axes[0].set_xlabel('Matrix Entry')

        # Right subplot: Frobenius norm histogram
        frob_norms = np.linalg.norm(connectivity_blocks, axis=(1, 2))  # Frobenius norm for each ii
        axes[1].hist(frob_norms, bins=80, color='C0', alpha=0.7)
        axes[1].set_title(f'Layer {self.layer_idx}: Distribution of Frobenius Norms of J_{{ij}}')
        axes[1].set_xlabel('Frobenius Norm')
        axes[1].set_ylabel('Count')
        
        plt.tight_layout()
        if show:
            plt.show()
    
    def plot_Frob_norms_summary(self, sum_wrt: str = 'kernel', show: bool = True) -> None:
        """Plot summary statistics of connectivity matrices."""
        if not sum_wrt in ['kernel', 'channels']:
            raise ValueError('sum_wrt must be either "kernel" or "channels".')

        if self.connectivity_weight is None:
            self.extract_connectivity_weights()
            
        J = self.connectivity_weight['weight']
        C_out, C_in, H, W = J.shape
        
        # Compute Frobenius norms of 2x2 blocks
        frobenius_maps = np.zeros((C_out//2, C_in//2, H, W))
        for k in range(C_out//2):
            for l in range(C_in//2):
                for i in range(H):
                    for j in range(W):
                        block = J[2*k:2*k+2, 2*l:2*l+2, i, j]
                        frobenius_maps[k, l, i, j] = np.linalg.norm(block, ord='fro')
        
        # Compute statistics
        if sum_wrt == 'kernel':            
            mean_grid = frobenius_maps.mean(axis=(0,1))
            std_grid = frobenius_maps.std(axis=(0,1))
            min_grid = frobenius_maps.min(axis=(0,1))
            max_grid = frobenius_maps.max(axis=(0,1))
        elif sum_wrt == 'channels':
            mean_grid = frobenius_maps.mean(axis=(2,3))
            std_grid = frobenius_maps.std(axis=(2,3))
            min_grid = frobenius_maps.min(axis=(2,3))
            max_grid = frobenius_maps.max(axis=(2,3))
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        stats = [mean_grid, std_grid, min_grid, max_grid]
        titles = ['Mean', 'Std', 'Min', 'Max']
        cmaps = ['viridis', 'magma', 'Blues', 'Reds']
        
        for ax, stat, title, cmap in zip(axes.flat, stats, titles, cmaps):
            im = ax.imshow(stat, cmap=cmap)
            ax.set_title(f'{title} of Frobenius Norms')
            ax.set_xlabel('Kernel W')
            ax.set_ylabel('Kernel H')
            ax.grid(False)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Layer {self.layer_idx} Connectivity Statistics', fontsize=16)
        plt.tight_layout()
        if show:
            plt.show()
    
    def plot_decomposition_results(self, decomp_type: str = 'symmetric_skew', show: bool = True) -> None:
        """Plot results of connectivity decomposition."""
        # Compute fresh decomposition for the specified layer
        connectivity_blocks = self.extract_connectivity_blocks()
        
        if decomp_type == 'symmetric_skew':
            p1, p2, p3, q, sym_frob, skew_frob = self.decompose_symmetric_skew(connectivity_blocks)
            results = {'p1': p1, 'p2': p2, 'p3': p3, 'q': q, 'sym_frob': sym_frob, 'skew_frob': skew_frob}
        elif decomp_type == 'rotation_symmetric':
            c_R, c_S, alpha, beta = self.decompose_rotation_symmetric(connectivity_blocks)
            results = {'c_R': c_R, 'c_S': c_S, 'alpha': alpha, 'beta': beta}
        else:
            raise ValueError(f"Decomposition type {decomp_type} not available.")
        
        if decomp_type == 'symmetric_skew':
            # Plot symmetric/skew components
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            components = ['p1', 'p2', 'p3', 'q']
            component_names = ['Jsym11', 'Jsym12', 'Jsym22', 'Jskew21']
            
            for i, comp in enumerate(components):
                data = results[comp]
                axes[i].hist(data, bins=50, alpha=0.7)
                axes[i].set_title(f'{component_names[i]} Distribution')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Count')
            
            plt.suptitle(f'Layer {self.layer_idx}: Symmetric/Skew-Symmetric Decomposition')
            plt.tight_layout()
            if show:
                plt.show()

            # Plot Frobenius norms of symmetry & asymmetry parts
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # violin plot of these Frobenius norms
            df_frob = pd.DataFrame({
                'Symmetric': results['sym_frob'],
                'Skew-Symmetric': results['skew_frob']
            })
            sns.violinplot(data=df_frob, ax=axes[0], inner='quartile')
            axes[0].set_title('Frobenius Norms: Symmetric vs Skew-Symmetric')
            axes[0].set_ylabel('Frobenius Norm')

            # scatter plot of the Frobenius norms
            axes[1].scatter(results['sym_frob'], results['skew_frob'], alpha=0.05, s=4)
            axes[1].set_xlabel('Symmetric Frobenius Norm')
            axes[1].set_ylabel('Skew-Symmetric Frobenius Norm')
            axes[1].set_title('Symmetric vs Skew-Symmetric Strength')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            if show:
                plt.show()

        elif decomp_type == 'rotation_symmetric':
            # Plot rotation/symmetric components
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            
            c_R = results['c_R']
            c_S = results['c_S']
            alpha = results['alpha']
            beta = results['beta']
            
            # Magnitude scatter plot
            axes[0,1].scatter(c_R, c_S, alpha=0.05, s=4)
            axes[0,1].set_xlabel('$c_R$')
            axes[0,1].set_ylabel('$c_S$')
            axes[0,1].set_title('Rotation vs Symmetric Strength')
            
            # Angle scatter plot
            axes[1,1].scatter(alpha, beta, alpha=0.05, s=4)
            axes[1,1].set_xlabel('$\\alpha$')
            axes[1,1].set_ylabel('$\\beta$')
            axes[1,1].set_title('Rotation vs Symmetric Angles')
            
            # Magnitude distributions
            df_magnitudes = pd.DataFrame({'c_R': c_R, 'c_S': c_S})
            sns.violinplot(data=df_magnitudes, ax=axes[0,0], inner='quartile')
            axes[0,0].set_title('Magnitude Distributions')
            
            # Angle distributions
            df_angles = pd.DataFrame({'alpha': alpha, 'beta': beta})
            sns.violinplot(data=df_angles, ax=axes[1,0], inner='quartile')
            axes[1,0].set_title('Angle Distributions')
            
            plt.tight_layout()
            if show:
                plt.show()
    
    def plot_clustering_results(self, show: bool = True) -> None:
        """Plot connectivity clustering results."""
        if self.clustering_results is None:
            self.analyze_connectivity_clustering()
            
        results = self.clustering_results
        
        # Plot silhouette scores
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].plot(results['k_range'], results['silhouette_scores'], 'o-')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Silhouette Score')
        axes[0].set_title('Clustering Quality vs k')
        axes[0].grid(True, alpha=0.3)
        
        # Plot cluster centers
        centers = results['centers']
        n_clusters = len(centers)
        
        # Show cluster centers as heatmaps
        for i in range(min(n_clusters, 8)):  # Show up to 8 clusters
            row = i // 4
            col = i % 4
            if i == 0:
                axes[1].imshow(centers[i], cmap='RdBu_r')
                axes[1].set_title(f'Cluster {i} Center')
            else:
                # Add more subplots if needed
                pass
        
        plt.tight_layout()
        if show:
            plt.show()
    
    def plot_dimensionality_reduction(self, method: str = 'pca', show: bool = True) -> None:
        """Plot dimensionality reduction results."""
        if self.dimensionality_reduction_results is None:
            self.perform_dimensionality_reduction()
            
        results = self.dimensionality_reduction_results
        
        if method not in results:
            raise ValueError(f"Method {method} not available. Available: {list(results.keys())}")
        
        embedding = results[method]['embedding']
        
        if method == 'pca':
            # 2D and 3D plots for PCA
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 2D plot
            axes[0].scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, s=5)
            axes[0].set_xlabel('PC1')
            axes[0].set_ylabel('PC2')
            axes[0].set_title('PCA 2D Visualization')
            
            # 3D plot
            ax = fig.add_subplot(122, projection='3d')
            ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], alpha=0.5, s=5)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.set_title('PCA 3D Visualization')
            
            plt.tight_layout()
            if show:
                plt.show()
            
        else:
            # 2D plot for t-SNE/UMAP
            plt.figure(figsize=(8, 6))
            plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, s=5)
            plt.xlabel(f'{method.upper()}-1')
            plt.ylabel(f'{method.upper()}-2')
            plt.title(f'{method.upper()} Visualization')
            plt.tight_layout()
            if show:
                plt.show()
    
    def plot_torus_visualization(self, show: bool = True) -> None:
        """Plot connectivity angles on a torus."""
        # Compute fresh decomposition for the specified layer
        connectivity_blocks = self.extract_connectivity_blocks()
        c_R, c_S, alpha, beta = self.decompose_rotation_symmetric(connectivity_blocks)
        
        # Torus parameters
        R, r = 3.0, 0.5
        
        # Create torus surface
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, 2 * np.pi, 40)
        U, V = np.meshgrid(u, v)
        X_surf = (R + r * np.cos(V)) * np.cos(U)
        Y_surf = (R + r * np.cos(V)) * np.sin(U)
        Z_surf = r * np.sin(V)
        
        # Normalize angles
        alpha_mod = np.mod(alpha, 2 * np.pi)
        beta_mod = np.mod(beta, 2 * np.pi)
        
        # Map data points to torus
        X = (R + r * np.cos(beta_mod)) * np.cos(alpha_mod)
        Y = (R + r * np.cos(beta_mod)) * np.sin(alpha_mod)
        Z = r * np.sin(beta_mod)
        
        # Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot torus surface
        ax.plot_surface(X_surf, Y_surf, Z_surf, color='wheat', alpha=0.7, 
                       edgecolor='none', shade=True)
        
        # Overlay data points
        scatter = ax.scatter(X, Y, Z, c=alpha_mod, cmap='twilight', s=6, alpha=0.7)
        
        ax.set_box_aspect([1, 1, 0.5])
        ax.set_axis_off()
        ax.view_init(elev=30, azim=45)
        ax.set_title(f'Layer {self.layer_idx}: Distribution of (α, β) on Torus', pad=20, fontsize=16)
        
        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        plt.tight_layout()
        if show:
            plt.show()

    def show_full_basic_plots(self) -> None:
        """
        Display all available plots for comprehensive analysis.
        """
        print(f"Generating all plots for layer {self.layer_idx}...")
        
        # 1. Plot omega distributions
        print("1. Omega parameter distributions...")
        self.plot_omega_distributions()
        
        # 2. Plot connectivity statistics
        print("2. Connectivity statistics...")
        self.plot_connectivity_statistics()
        
        # 3. Plot Frobenius norms summary (kernel)
        print("3. Frobenius norms summary (kernel)...")
        self.plot_Frob_norms_summary(sum_wrt='kernel')
        
        # 4. Plot Frobenius norms summary (channels)
        print("4. Frobenius norms summary (channels)...")
        self.plot_Frob_norms_summary(sum_wrt='channels')
        
        # 5. Plot decomposition results (symmetric_skew)
        print("5. Symmetric/Skew-Symmetric decomposition...")
        self.plot_decomposition_results(decomp_type='symmetric_skew')
        
        # 6. Plot decomposition results (rotation_symmetric)
        print("6. Rotation/Symmetric decomposition...")
        self.plot_decomposition_results(decomp_type='rotation_symmetric')
        
        # 7. Plot clustering results (if available)
        print("7. Clustering results...")
        print(f"   Clustering analysis omitted for now")
        # try:
        #     self.plot_clustering_results()
        # except Exception as e:
        #     print(f"   Clustering analysis not available: {e}")
        
        # 8. Plot dimensionality reduction (PCA)
        print("8. Dimensionality reduction (PCA)...")
        try:
            self.plot_dimensionality_reduction(method='pca')
        except Exception as e:
            print(f"   PCA visualization not available: {e}")
       
        print(f"All basic plots completed for layer {self.layer_idx}!")
    
    def generate_analysis_report(self, save_path: Optional[str] = None) -> Dict:
        """
        Generate a comprehensive analysis report.
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Dictionary containing the full analysis report
        """
        print(f"Generating comprehensive analysis report for layer {self.layer_idx}...")
        
        # Extract all basic parameters
        self.extract_omega_parameters()
        self.extract_connectivity_weights()
        
        # Perform full connectivity analysis
        full_analysis = self.analyze_full_connectivity()
        
        # Compile report
        report = {
            'model_info': {
                'num_layers': self.num_layers,
                'oscillator_dimension': self.n,
                'analyzed_layer': self.layer_idx
            },
            'omega_analysis': {
                'has_omega': self.omega_param is not None,
                'omega_shape': self.omega_param.shape if self.omega_param is not None else None,
                'omega_magnitude': float(np.linalg.norm(self.omega_param)) if self.omega_param is not None else None
            },
            'connectivity_analysis': {
                'layer_shape': self.connectivity_weight['shape'] if self.connectivity_weight else None,
                'weight_statistics': {
                    'mean': float(self.connectivity_weight['weight'].mean()),
                    'std': float(self.connectivity_weight['weight'].std()),
                    'min': float(self.connectivity_weight['weight'].min()),
                    'max': float(self.connectivity_weight['weight'].max())
                } if self.connectivity_weight else None,
                'num_blocks': len(self.connectivity_blocks) if self.connectivity_blocks is not None else None,
                'block_statistics': {
                    'frobenius_norms': {
                        'mean': float(np.linalg.norm(self.connectivity_blocks, axis=(1,2)).mean()),
                        'std': float(np.linalg.norm(self.connectivity_blocks, axis=(1,2)).std())
                    }
                } if self.connectivity_blocks is not None else None
            },
            'decomposition_analysis': {
                'symmetric_skew_stats': {
                    comp: {
                        'mean': float(vals.mean()),
                        'std': float(vals.std())
                    }
                    for comp, vals in full_analysis['decomposition']['symmetric_skew'].items()
                    if isinstance(vals, np.ndarray)
                },
                'rotation_symmetric_stats': {
                    comp: {
                        'mean': float(vals.mean()),
                        'std': float(vals.std())
                    }
                    for comp, vals in full_analysis['decomposition']['rotation_symmetric'].items()
                    if isinstance(vals, np.ndarray)
                }
            } if full_analysis['decomposition'] else None,
            'clustering_analysis': {
                'optimal_k': full_analysis['clustering']['best_k'],
                'silhouette_score': full_analysis['clustering']['best_score'],
                'cluster_sizes': [
                    int(np.sum(full_analysis['clustering']['labels'] == i))
                    for i in range(full_analysis['clustering']['best_k'])
                ]
            } if full_analysis['clustering'] else None
        }
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {save_path}")
            
        return report
