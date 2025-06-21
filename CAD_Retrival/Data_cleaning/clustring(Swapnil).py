import os
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from PIL import Image
import logging
from tqdm import tqdm
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageClusterer:
    def __init__(self, model_name="ViT-B/32", num_clusters=5, pca_components=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()  # Ensure model is in evaluation mode
        
        self.num_clusters = num_clusters
        self.pca_components = pca_components
        self.scaler = StandardScaler()
        
    def load_images(self, image_folder):
        """Load and validate images from folder."""
        image_folder = Path(image_folder)
        valid_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        
        image_paths = []
        for ext in valid_extensions:
            image_paths.extend(list(image_folder.glob(f"*{ext}")))
            
        if not image_paths:
            raise ValueError(f"No valid images found in {image_folder}")
            
        logger.info(f"Found {len(image_paths)} images")
        return image_paths
    
    def extract_features(self, image_paths):
        """Extract CLIP features with error handling."""
        features = []
        valid_paths = []
        
        for img_path in tqdm(image_paths, desc="Extracting Features"):
            try:
                # Handle RGBA images
                image = Image.open(img_path)
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                
                # Process image and extract features
                with torch.no_grad():
                    image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                    feature = self.model.encode_image(image_input)
                    features.append(feature.cpu().numpy().flatten())
                    valid_paths.append(img_path)
                    
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {str(e)}")
                continue
                
        if not features:
            raise ValueError("No valid features extracted from images")
            
        return np.array(features), valid_paths
    
    def reduce_dimensions(self, features):
        """Reduce dimensions with PCA and scaling."""
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA
        pca = PCA(n_components=min(self.pca_components, features.shape[0], features.shape[1]))
        features_pca = pca.fit_transform(features_scaled)
        
        explained_variance = np.sum(pca.explained_variance_ratio_)
        logger.info(f"Explained variance after PCA: {explained_variance:.2f}")
        
        return features_pca
    
    def cluster_images(self, features_pca):
        """Perform K-means clustering with stability check."""
        # Determine optimal number of clusters if not specified
        if self.num_clusters > len(features_pca):
            logger.warning("Reducing number of clusters to match dataset size")
            self.num_clusters = min(self.num_clusters, len(features_pca))
        
        kmeans = KMeans(n_clusters=self.num_clusters, 
                       random_state=42, 
                       n_init=10)  # Multiple initializations for stability
        labels = kmeans.fit_predict(features_pca)
        
        # Check cluster sizes
        cluster_sizes = np.bincount(labels)
        logger.info(f"Cluster sizes: {cluster_sizes}")
        
        return labels, kmeans
    
    def visualize_clusters(self, features_pca, labels):
        """Create t-SNE visualization with improved aesthetics."""
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_pca)-1))
        features_tsne = tsne.fit_transform(features_pca)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                            c=labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title("Image Clusters Visualization")
        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        return plt.gcf()
    
    def save_clusters(self, output_folder, image_paths, labels):
        """Save clustered images with metadata."""
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Save cluster metadata
        metadata = {}
        for i in range(self.num_clusters):
            cluster_folder = output_folder / f'cluster_{i}'
            cluster_folder.mkdir(exist_ok=True)
            
            cluster_indices = np.where(labels == i)[0]
            metadata[f'cluster_{i}'] = {
                'size': len(cluster_indices),
                'images': []
            }
            
            for idx in cluster_indices:
                src_path = image_paths[idx]
                dst_path = cluster_folder / src_path.name
                
                try:
                    img = cv2.imread(str(src_path))
                    if img is None:
                        raise ValueError("Failed to read image")
                    cv2.imwrite(str(dst_path), img)
                    metadata[f'cluster_{i}']['images'].append(src_path.name)
                except Exception as e:
                    logger.error(f"Error saving {src_path}: {str(e)}")
                    
        # Save metadata
        np.save(output_folder / 'cluster_metadata.npy', metadata)
        
    def run(self, input_folder, output_folder):
        """Run the complete clustering pipeline."""
        try:
            # Load and process images
            image_paths = self.load_images(input_folder)
            features, valid_paths = self.extract_features(image_paths)
            
            # Reduce dimensions and cluster
            features_pca = self.reduce_dimensions(features)
            labels, kmeans = self.cluster_images(features_pca)
            
            # Visualize results
            fig = self.visualize_clusters(features_pca, labels)
            
            # Save results
            self.save_clusters(output_folder, valid_paths, labels)
            fig.savefig(Path(output_folder) / 'cluster_visualization.png')
            
            logger.info("Clustering completed successfully!")
            return labels, kmeans
            
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            raise

# Usage example
if __name__ == "__main__":
    input_folder = "/media/swapnil/3f73cc1a-8f9d-4c19-87af-99b3512ff5b2/04Joints/clusters/cluster_0"
    output_folder = "/media/swapnil/3f73cc1a-8f9d-4c19-87af-99b3512ff5b2/04Joints/output"
    
    clusterer = ImageClusterer(num_clusters=5)
    labels, kmeans = clusterer.run(input_folder, output_folder)