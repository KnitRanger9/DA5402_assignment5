import os
import logging
import random
import shutil
import subprocess

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cifar10_partitioning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_partitions(seed=42):
    """
    Create three partitions of 20,000 images from the original dataset
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Source directory with original images
    source_dir = 'data/cifar10_images'
    
    # Get all image paths
    all_images = []
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.png'):
                    all_images.append((class_name, img_name))
    
    logger.info(f"Total images found: {len(all_images)}")
    
    # Shuffle all images
    random.shuffle(all_images)
    
    # Create three partitions
    partitions = [
        all_images[:20000],
        all_images[20000:40000],
        all_images[40000:60000]
    ]
    
    # Create directories for partitions
    for i, partition in enumerate(partitions):
        partition_dir = f'partitions/partition_v{i}/data'
        if os.path.exists(partition_dir):
            shutil.rmtree(partition_dir)
        os.makedirs(partition_dir)
        logger.info(f"Created directory: {partition_dir}")
        
        # Create class directories
        class_counts = {}
        for class_name in os.listdir(source_dir):
            class_dir = os.path.join(partition_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            class_counts[class_name] = 0
        
        # Copy images to partition
        for class_name, img_name in partition:
            src_path = os.path.join(source_dir, class_name, img_name)
            dst_path = os.path.join(partition_dir, class_name, img_name)
            shutil.copy2(src_path, dst_path)
            class_counts[class_name] += 1
        
        # Log class distribution
        logger.info(f"Partition v{i} class distribution:")
        for class_name, count in class_counts.items():
            logger.info(f"  {class_name}: {count} images")
        logger.info(f"Total images in partition v{i}: {len(partition)}")
    
    logger.info("All partitions created successfully")

def setup_git_repo(repo_name="cifar10-partitions", remote_url=None):
    """
    Set up Git repository
    """
    try:
        # Initialize Git repository if it doesn't exist
        if not os.path.exists('.git'):
            subprocess.run(['git', 'init'], check=True)
            logger.info("Git repository initialized")
            
            # Configure Git (optional)
            subprocess.run(['git', 'config', 'user.name', 'KnitRanger9'], check=True)
            subprocess.run(['git', 'config', 'user.email', 'cifar10@example.com'], check=True)
            
        # Add remote if provided
        if remote_url:
            subprocess.run(['git', 'remote', 'add', 'origin', remote_url], check=True)
            logger.info(f"Added remote: {remote_url}")
            
        # Create .gitignore
        with open('.gitignore', 'w') as f:
            f.write("*.log\n")
            f.write("__pycache__/\n")
            f.write("*.pyc\n")
            f.write("data/cifar10_images/\n")  # Don't track original dataset
            
        # Add .gitignore to Git
        # subprocess.run(['git', 'add', '.gitignore'], check=True)
        # subprocess.run(['git', 'commit', '-m', 'Initial commit with .gitignore'], check=True)
        # logger.info("Created .gitignore and committed")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during Git setup: {e}")

def push_to_dvc_and_git(remote_url=None):
    """
    Add partitions to DVC, tag them with Git, and push to remote
    """
    try:
        # Initialize DVC if not already initialized
        if not os.path.exists('.dvc'):
            subprocess.run(['dvc', 'init'], check=True)
            logger.info("DVC initialized")
            # Add .dvc directory to Git
            subprocess.run(['git', 'add', '.dvc'], check=True)
            subprocess.run(['git', 'commit', '-m', 'Initialize DVC'], check=True)
        
        # Add each partition to DVC
        for i in range(0, 3):
            partition_dir = f'partitions/partition_v{i}'
            
            # Add to DVC
            subprocess.run(['dvc', 'add', f"{partition_dir}/data"], check=True)
            logger.info(f"Added {partition_dir} to DVC")
            
            # Add .dvc file to Git
            subprocess.run(['git', 'add', f'{partition_dir}.dvc'], check=True)
            
            # Commit changes
            commit_msg = f'Add dataset partition v{i}'
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
            
            # Add Git tag instead of DVC tag
            tag_name = f'v{i}'
            tag_message = f'Dataset partition v{i}'
            subprocess.run(['git', 'tag', '-a', tag_name, '-m', tag_message], check=True)
            logger.info(f"Tagged {partition_dir} with Git tag {tag_name}")
        
        # Push to remote if URL is provided
        if remote_url:
            # Push commits
            subprocess.run(['git', 'push', 'origin', 'main'], check=True)
            # Push tags
            subprocess.run(['git', 'push', 'origin', '--tags'], check=True)
            logger.info("Pushed commits and tags to remote")
            
            # Push DVC data to remote if configured
            try:
                subprocess.run(['dvc', 'push'], check=True)
                logger.info("Pushed DVC data to remote storage")
            except subprocess.CalledProcessError:
                logger.warning("DVC remote storage not configured or push failed")
        
        logger.info("All partitions added to DVC and Git successfully")
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during DVC/Git operations: {e}")

if __name__ == "__main__":
    # Set up the Git repository first
    setup_git_repo(remote_url=None)  # Add your remote URL if needed
    
    # Create partitions
    create_partitions()
    
    # Add to DVC and Git, and push
    #push_to_dvc_and_git(remote_url=None)  # Add your remote URL if needed