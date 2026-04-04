#!/usr/bin/env python3
"""
PCA Feature Visualization Tool for Model Output Features

This script performs Principal Component Analysis (PCA) on model output features
and generates 2D/3D visualizations to help understand the feature space distribution.

Usage:
    python pca_visualizer.py --input <features.npy> [--output <output_dir>] 
                             [--n_components <2|3>] [--labels <labels.csv>]
                             [--title <plot_title>]

Example:
    # Basic 2D visualization
    python pca_visualizer.py --input features.npy --output viz_output
    
    # 3D visualization with labels
    python pca_visualizer.py --input features.npy --labels labels.csv \
                             --n_components 3 --output viz_output \
                             --title "Feature Space Visualization"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json


def load_features(input_path: str) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Load feature data from various file formats.
    
    Supported formats:
        - .npy: NumPy array format
        - .npz: Compressed NumPy arrays (expects 'features' key)
        - .csv: CSV file with numeric data
        - .txt: Text file with numeric data (space/tab separated)
    
    Args:
        input_path: Path to the input feature file
        
    Returns:
        Tuple of (feature_array, feature_names_if_available)
    """
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    feature_names = None
    
    if path.suffix.lower() == '.npy':
        data = np.load(input_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            print(f"Warning: Input data has {data.ndim} dimensions, flattening to 2D")
            data = data.reshape(data.shape[0], -1)
            
    elif path.suffix.lower() == '.npz':
        data_dict = np.load(input_path)
        if 'features' in data_dict:
            data = data_dict['features']
        elif len(data_dict.files) == 1:
            data = data_dict[data_dict.files[0]]
        else:
            raise ValueError("NPZ file must contain 'features' key or a single array")
            
    elif path.suffix.lower() == '.csv':
        try:
            import pandas as pd
            df = pd.read_csv(input_path)
            # Check if first row might be column names
            feature_names = list(df.columns) if all(isinstance(col, str) for col in df.columns) else None
            data = df.values
        except ImportError:
            # Fallback to numpy without pandas
            data = np.loadtxt(input_path, delimiter=',')
            feature_names = None
            
    elif path.suffix.lower() == '.txt':
        data = np.loadtxt(input_path)
        
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Supported: .npy, .npz, .csv, .txt")
    
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")
    
    print(f"Loaded features with shape: {data.shape}")
    return data, feature_names


def load_labels(labels_path: Optional[str], n_samples: int) -> Optional[np.ndarray]:
    """
    Load label data for coloring the visualization.
    
    Args:
        labels_path: Path to labels file (optional)
        n_samples: Number of samples expected
        
    Returns:
        Label array or None if no labels provided
    """
    if labels_path is None:
        return None
    
    path = Path(labels_path)
    if not path.exists():
        print(f"Warning: Labels file not found: {labels_path}")
        return None
    
    try:
        import pandas as pd
        labels_df = pd.read_csv(labels_path)
        # Get the first column or 'label' column if exists
        if 'label' in labels_df.columns:
            labels = labels_df['label'].values
        elif 'labels' in labels_df.columns:
            labels = labels_df['labels'].values
        else:
            labels = labels_df.iloc[:, 0].values
    except ImportError:
        # Fallback to numpy
        labels = np.loadtxt(labels_path, delimiter=',', ndmin=1)
        if labels.ndim > 1:
            labels = labels[:, 0]
    
    if len(labels) != n_samples:
        print(f"Warning: Number of labels ({len(labels)}) doesn't match number of samples ({n_samples})")
        return None
    
    print(f"Loaded {len(np.unique(labels))} unique labels")
    return labels


def perform_pca(features: np.ndarray, n_components: int = 2, 
                standardize: bool = True) -> Tuple[np.ndarray, PCA, Dict[str, float]]:
    """
    Perform PCA on the feature data.
    
    Args:
        features: Input feature array of shape (n_samples, n_features)
        n_components: Number of principal components (2 or 3)
        standardize: Whether to standardize features before PCA
        
    Returns:
        Tuple of (transformed_data, pca_model, explained_variance_ratio)
    """
    print(f"\nPerforming PCA with {n_components} components...")
    
    # Handle missing values
    if np.any(np.isnan(features)):
        print("Warning: NaN values detected, replacing with column means")
        col_means = np.nanmean(features, axis=0)
        inds = np.where(np.isnan(features))
        features[inds] = np.take(col_means, inds[1])
    
    # Standardize features
    if standardize:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        print("Features standardized (zero mean, unit variance)")
    else:
        features_scaled = features
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(features_scaled)
    
    # Calculate statistics
    stats = {
        'explained_variance_ratio': float(pca.explained_variance_ratio_.sum()),
        'individual_variance_ratios': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance_ratio': float(np.cumsum(pca.explained_variance_ratio_)[-1]),
        'n_original_features': features.shape[1],
        'n_samples': features.shape[0]
    }
    
    print(f"\nPCA Results:")
    print(f"  Original feature dimension: {stats['n_original_features']}")
    print(f"  Reduced dimension: {n_components}")
    print(f"  Explained variance ratio (PC1): {pca.explained_variance_ratio_[0]:.4f}")
    print(f"  Explained variance ratio (PC2): {pca.explained_variance_ratio_[1]:.4f}")
    if n_components == 3:
        print(f"  Explained variance ratio (PC3): {pca.explained_variance_ratio_[2]:.4f}")
    print(f"  Total explained variance: {stats['explained_variance_ratio']:.4f}")
    
    return transformed, pca, stats


def create_2d_plot(transformed: np.ndarray, labels: Optional[np.ndarray],
                   output_path: str, title: str, 
                   variance_info: Dict[str, float]) -> None:
    """
    Create a 2D scatter plot of the PCA results.
    
    Args:
        transformed: Transformed data of shape (n_samples, 2)
        labels: Optional labels for coloring
        output_path: Path to save the plot
        title: Plot title
        variance_info: Dictionary containing variance information
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    pc1 = transformed[:, 0]
    pc2 = transformed[:, 1]
    
    if labels is not None:
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)
        
        # Choose colormap based on number of labels
        if n_labels <= 10:
            cmap = plt.get_cmap('tab10')
        elif n_labels <= 20:
            cmap = plt.get_cmap('tab20')
        else:
            cmap = plt.get_cmap('viridis')
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(pc1[mask], pc2[mask], 
                      c=[cmap(i % cmap.N)], 
                      label=f'Class {label}',
                      alpha=0.6,
                      edgecolors='w',
                      s=50)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    else:
        scatter = ax.scatter(pc1, pc2, 
                            c=pc1, 
                            cmap='viridis',
                            alpha=0.6,
                            edgecolors='w',
                            s=50)
        plt.colorbar(scatter, ax=ax, label='PC1 value')
    
    # Add variance information to the plot
    xlabel = f'PC1 ({variance_info["individual_variance_ratios"][0]:.2%} variance)'
    ylabel = f'PC2 ({variance_info["individual_variance_ratios"][1]:.2%} variance)'
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add text box with summary statistics
    textstr = f'Total Variance: {variance_info["explained_variance_ratio"]:.2%}\nSamples: {variance_info["n_samples"]}\nOriginal Dim: {variance_info["n_original_features"]}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"2D plot saved to: {output_path}")


def create_3d_plot(transformed: np.ndarray, labels: Optional[np.ndarray],
                   output_path: str, title: str,
                   variance_info: Dict[str, float]) -> None:
    """
    Create a 3D scatter plot of the PCA results.
    
    Args:
        transformed: Transformed data of shape (n_samples, 3)
        labels: Optional labels for coloring
        output_path: Path to save the plot
        title: Plot title
        variance_info: Dictionary containing variance information
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    pc1 = transformed[:, 0]
    pc2 = transformed[:, 1]
    pc3 = transformed[:, 2]
    
    if labels is not None:
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)
        
        if n_labels <= 10:
            cmap = plt.get_cmap('tab10')
        elif n_labels <= 20:
            cmap = plt.get_cmap('tab20')
        else:
            cmap = plt.get_cmap('viridis')
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(pc1[mask], pc2[mask], pc3[mask],
                      c=[cmap(i % cmap.N)],
                      label=f'Class {label}',
                      alpha=0.6,
                      edgecolors='w',
                      s=50)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    else:
        scatter = ax.scatter(pc1, pc2, pc3,
                            c=pc1,
                            cmap='viridis',
                            alpha=0.6,
                            edgecolors='w',
                            s=50)
        plt.colorbar(scatter, ax=ax, label='PC1 value', shrink=0.5)
    
    xlabel = f'PC1 ({variance_info["individual_variance_ratios"][0]:.2%})'
    ylabel = f'PC2 ({variance_info["individual_variance_ratios"][1]:.2%})'
    zlabel = f'PC3 ({variance_info["individual_variance_ratios"][2]:.2%})'
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_zlabel(zlabel, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add text box with summary statistics
    textstr = f'Total Variance: {variance_info["explained_variance_ratio"]:.2%}\nSamples: {variance_info["n_samples"]}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text2D(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"3D plot saved to: {output_path}")


def save_statistics(stats: Dict[str, Any], output_path: str) -> None:
    """Save PCA statistics to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='PCA Feature Visualization Tool for Model Output Features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic 2D visualization
  python pca_visualizer.py --input features.npy --output ./viz_output
  
  # 3D visualization with custom title
  python pca_visualizer.py --input features.npy --n_components 3 \\
                           --output ./viz_output --title "My Model Features"
  
  # With labels for class-based coloring
  python pca_visualizer.py --input features.npy --labels labels.csv \\
                           --output ./viz_output
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input feature file (.npy, .npz, .csv, .txt)')
    parser.add_argument('--output', '-o', default='./pca_output',
                       help='Output directory for plots and statistics (default: ./pca_output)')
    parser.add_argument('--n_components', '-n', type=int, default=2, choices=[2, 3],
                       help='Number of principal components for visualization (default: 2)')
    parser.add_argument('--labels', '-l', type=str, default=None,
                       help='Optional labels file for coloring (.csv, .txt)')
    parser.add_argument('--title', '-t', type=str, default='PCA Feature Visualization',
                       help='Title for the plot (default: "PCA Feature Visualization")')
    parser.add_argument('--no_standardize', action='store_true',
                       help='Disable feature standardization before PCA')
    parser.add_argument('--save_transformed', action='store_true',
                       help='Save transformed features to output directory')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PCA Feature Visualization Tool")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    try:
        # Load features
        print(f"\nLoading features from: {args.input}")
        features, feature_names = load_features(args.input)
        
        # Load labels if provided
        labels = load_labels(args.labels, features.shape[0])
        
        # Perform PCA
        transformed, pca_model, stats = perform_pca(
            features, 
            n_components=args.n_components,
            standardize=not args.no_standardize
        )
        
        # Create visualization
        if args.n_components == 2:
            plot_path = output_dir / 'pca_2d.png'
            create_2d_plot(transformed, labels, str(plot_path), args.title, stats)
        else:
            plot_path = output_dir / 'pca_3d.png'
            create_3d_plot(transformed, labels, str(plot_path), args.title, stats)
        
        # Save statistics
        stats_path = output_dir / 'pca_statistics.json'
        save_statistics(stats, str(stats_path))
        
        # Save transformed features if requested
        if args.save_transformed:
            transformed_path = output_dir / 'transformed_features.npy'
            np.save(str(transformed_path), transformed)
            print(f"Transformed features saved to: {transformed_path}")
        
        print("\n" + "=" * 60)
        print("Visualization complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
