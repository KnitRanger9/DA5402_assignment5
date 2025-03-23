import os
import yaml
import logging
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import imageio.v2 as imageio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/prepare_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_data():
    """Prepare train, validation, and test datasets"""
    # Load configuration
    config = load_config()
    random_seed = config['random_seed']
    train_ratio = config['dataset']['split']['train']
    val_ratio = config['dataset']['split']['val']
    test_ratio = config['dataset']['split']['test']
    
    # Verify split ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Split ratios must sum to 1"
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Load dataset
    with open('data/dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    image_paths = dataset['image_paths']
    labels = dataset['labels']
    class_names = dataset['class_names']
    
    logger.info(f"Preparing data splits. Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    
    # Split data into train and temporary datasets
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, train_size=train_ratio, 
        stratify=labels, random_state=random_seed
    )
    
    # Split temporary dataset into validation and test datasets
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, train_size=val_ratio_adjusted,
        stratify=temp_labels, random_state=random_seed
    )
    
    logger.info(f"Dataset split complete. Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Load images and preprocess
    def load_and_preprocess_images(paths, labels):
        """Load and preprocess images from paths"""
        images = []
        for path in paths:
            img = imageio.imread(path)
            # Convert to RGB if grayscale
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            images.append(img)
        return np.array(images), np.array(labels)
    
    # Process datasets
    logger.info("Loading and preprocessing train images...")
    X_train, y_train = load_and_preprocess_images(train_paths, train_labels)
    
    logger.info("Loading and preprocessing validation images...")
    X_val, y_val = load_and_preprocess_images(val_paths, val_labels)
    
    logger.info("Loading and preprocessing test images...")
    X_test, y_test = load_and_preprocess_images(test_paths, test_labels)
    
    # Create dataset dictionaries
    train_data = {
        'images': X_train,
        'labels': y_train,
        'paths': train_paths,
        'class_names': class_names
    }
    
    val_data = {
        'images': X_val,
        'labels': y_val,
        'paths': val_paths,
        'class_names': class_names
    }
    
    test_data = {
        'images': X_test,
        'labels': y_test,
        'paths': test_paths,
        'class_names': class_names
    }
    
    # Save datasets
    with open('data/train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    
    with open('data/val_data.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    
    with open('data/test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    
    logger.info("Data preparation completed successfully")
    
    # Log class distribution in splits
    for name, labels in [('Train', y_train), ('Validation', y_val), ('Test', y_test)]:
        unique_labels, counts = np.unique(labels, return_counts=True)
        logger.info(f"{name} class distribution:")
        for i, class_idx in enumerate(unique_labels):
            logger.info(f"  Class {class_names[class_idx]}: {counts[i]} images")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Prepare data
    prepare_data()