import os
import logging
import numpy as np
from tensorflow.keras.datasets import cifar10
import imageio.v2 as imageio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cifar10_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def download_and_organize_cifar10():
    """
    Download CIFAR-10 dataset and organize images into class folders
    """
    # Create data directory if it doesn't exist
    data_dir = 'data/cifar10_images'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info(f"Created directory: {data_dir}")
    
    # Download CIFAR-10 dataset
    logger.info("Downloading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Combine training and test data
    images = np.vstack([x_train, x_test])
    labels = np.vstack([y_train, y_test]).flatten()
    
    logger.info(f"Total images: {len(images)}")
    
    # Create class directories and save images
    for class_idx, class_name in enumerate(class_names):
        # Create class directory
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            logger.info(f"Created directory: {class_dir}")
        
        # Get indices of images belonging to current class
        indices = np.where(labels == class_idx)[0]
        logger.info(f"Class {class_name}: {len(indices)} images")
        
        # Save images to class directory
        for i, idx in enumerate(indices):
            img_path = os.path.join(class_dir, f"{class_name}_{i}.png")
            imageio.imwrite(img_path, images[idx])
        
        logger.info(f"Saved {len(indices)} images for class {class_name}")
    
    logger.info("CIFAR-10 dataset organized successfully")

if __name__ == "__main__":
    download_and_organize_cifar10()