import os
import yaml
import logging
import pickle
import numpy as np
import subprocess
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/pull_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def pull_data():
    """Pull the specified version of the dataset from DVC"""
    # Load configuration
    config = load_config()
    version = config['dataset']['version']
    random_seed = config['random_seed']
    
    # Set random seed
    np.random.seed(random_seed)
    
    logger.info(f"Pulling dataset version {version} from DVC")
    
    # Pull the specified version
    try:
        subprocess.run(['dvc', 'checkout', f'partitions/partition_{version}/data.dvc'], check=True)
        logger.info(f"Successfully pulled dataset version {version}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error pulling dataset: {e}")
        raise
    
    # Ensure output directory exists
    os.makedirs('data', exist_ok=True)
    
    # Process the images and convert to a dataframe-like structure
    partition_dir = f'partitions/partition_{version}/data'
    
    # Initialize lists to store data
    images = []
    labels = []
    image_paths = []
    
    # Get class names
    class_names = sorted([d for d in os.listdir(partition_dir) if os.path.isdir(os.path.join(partition_dir, d))])
    logger.info(f"Found classes: {class_names}")
    
    # Process each class
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(partition_dir, class_name)
        logger.info(f"Processing class {class_name} ({class_idx})")
        
        # Process each image in the class
        for img_name in os.listdir(class_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                # Save the image path
                image_paths.append(img_path)
                # Save the label
                labels.append(class_idx)
    
    # Store dataset information
    dataset = {
        'version': version,
        'image_paths': image_paths,
        'labels': labels,
        'class_names': class_names
    }
    
    # Save dataset
    with open('data/dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    
    logger.info(f"Dataset processed and saved. Total images: {len(image_paths)}")
    
    # Log class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    for i, class_idx in enumerate(unique_labels):
        logger.info(f"Class {class_names[class_idx]}: {counts[i]} images")
    
    return dataset

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Pull data
    pull_data()