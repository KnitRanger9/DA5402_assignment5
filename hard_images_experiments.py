import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/hard_images.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_predictions(version, seed):
    """Load image predictions for a specific experiment"""
    # Construct the path to the predictions file
    pred_file = f"results/image_predictions.json"
    
    # Check if file exists
    if not os.path.exists(pred_file):
        logger.warning(f"Predictions file not found for version={version}, seed={seed}")
        return None
    
    # Load predictions
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    # Add experiment info to each prediction
    for pred in predictions:
        pred['dataset_version'] = version
        pred['seed'] = seed
    
    return predictions

def find_hard_images():
    """
    Identify images that are consistently misclassified across models
    trained on different dataset versions
    """
    # Create directory for analysis results
    os.makedirs('analysis/hard_images', exist_ok=True)
    
    # Versions to analyze
    versions = ['v1', 'v1v2', 'v1v2v3']
    seed = 42  # Using one seed for comparison
    
    # Load predictions for each version
    all_predictions = []
    for version in versions:
        preds = load_predictions(version, seed)
        if preds:
            all_predictions.extend(preds)
    
    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(all_predictions)
    
    # Focus on images from v1 partition
    v1_images = [path for path in df['image_path'].unique() if 'partition_v1' in path]
    logger.info(f"Found {len(v1_images)} images from v1 partition")
    
    # Create a dictionary to track misclassifications per image
    image_results = {}
    
    for img_path in v1_images:
        # Get predictions for this image across all models
        img_preds = df[df['image_path'] == img_path]
        
        # Track if the image was correctly classified by each model
        correct_by_model = {}
        for _, pred in img_preds.iterrows():
            version = pred['dataset_version']
            correct_by_model[version] = pred['correct']
        
        # If image was misclassified by all models, it's "hard to learn"
        is_hard = all(not correct for correct in correct_by_model.values())
        
        # Store results
        image_results[img_path] = {
            'path': img_path,
            'true_class': img_preds.iloc[0]['true_class'],
            'predictions': {
                version: {
                    'predicted_class': pred['predicted_class'],
                    'correct': pred['correct'],
                    'confidence': pred['confidence']
                }
                for version, pred in zip(img_preds['dataset_version'], img_preds.itertuples())
            },
            'is_hard': is_hard
        }
    
    # Filter hard images
    hard_images = {path: info for path, info in image_results.items() if info['is_hard']}
    logger.info(f"Found {len(hard_images)} hard-to-learn images")
    
    # Create class distribution of hard-to-learn images
    class_distribution = Counter([info['true_class'] for info in hard_images.values()])
    
    # Calculate the percentage of hard images per class
    class_counts = Counter([info['true_class'] for info in image_results.values()])
    class_percentages = {
        class_name: (class_distribution[class_name] / class_counts[class_name] * 100)
        for class_name in class_counts
    }
    
    # Create misclassification table
    misclassification_table = {}
    for info in hard_images.values():
        true_class = info['true_class']
        if true_class not in misclassification_table:
            misclassification_table[true_class] = Counter()
        
        # Count where this true class was misclassified
        for version in versions:
            if version in info['predictions']:
                pred_class = info['predictions'][version]['predicted_class']
                misclassification_table[true_class][pred_class] += 1
    
    # Convert to DataFrame for better visualization
    misclass_df = pd.DataFrame(misclassification_table).fillna(0)
    
    # Plot class distribution of hard images
    plt.figure(figsize=(10, 6))
    plt.bar(class_distribution.keys(), class_distribution.values())
    plt.title('Class Distribution of Hard-to-Learn Images')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analysis/hard_images/class_distribution.png', dpi=300)
    
    # Plot percentage of hard images per class
    plt.figure(figsize=(10, 6))
    plt.bar(class_percentages.keys(), class_percentages.values())
    plt.title('Percentage of Hard-to-Learn Images per Class')
    plt.xlabel('Class')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analysis/hard_images/class_percentages.png', dpi=300)
    
    # Plot misclassification heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(misclass_df, annot=True, cmap='Blues', fmt='g')
    plt.title('Misclassification Table for Hard-to-Learn Images')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig('analysis/hard_images/misclassification_heatmap.png', dpi=300)
    
    # Save hard image paths and info
    with open('analysis/hard_images/hard_images.json', 'w') as f:
        json.dump(hard_images, f, indent=2)
    
    # Create a visualization of some hard images
    visualize_hard_images(hard_images, versions)
    
    # Save misclassification table
    with open('analysis/hard_images/misclassification_table.json', 'w') as f:
        # Convert Counter objects to dictionaries for JSON serialization
        misclass_json = {
            true_class: dict(counter)
            for true_class, counter in misclassification_table.items()
        }
        json.dump(misclass_json, f, indent=2)
    
    # Save class distribution
    with open('analysis/hard_images/class_distribution.json', 'w') as f:
        json.dump(dict(class_distribution), f, indent=2)
    
    # Save class percentages
    with open('analysis/hard_images/class_percentages.json', 'w') as f:
        json.dump(class_percentages, f, indent=2)
    
    logger.info("Hard images analysis completed and saved to analysis/hard_images directory")
    
    return hard_images, class_distribution, misclassification_table

def visualize_hard_images(hard_images, versions, max_images=25):
    """Visualize a sample of hard-to-learn images with their predictions"""
    # Create directory for visualizations
    os.makedirs('analysis/hard_images/samples', exist_ok=True)
    
    # Sample images to visualize
    sample_paths = list(hard_images.keys())[:max_images]
    
    # Create a grid of images
    grid_size = min(5, len(sample_paths))
    fig, axes = plt.subplots(nrows=(len(sample_paths) + grid_size - 1) // grid_size, 
                            ncols=grid_size, 
                            figsize=(15, 3 * ((len(sample_paths) + grid_size - 1) // grid_size)))
    
    # Make axes a 2D array if it's 1D
    if len(sample_paths) <= grid_size:
        axes = np.array([axes])
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten()
    
    # Plot each image
    for i, (ax, img_path) in enumerate(zip(axes_flat, sample_paths)):
        if i < len(sample_paths):
            try:
                # Load image
                img = imageio.imread(img_path)
                
                # Display image
                ax.imshow(img)
                
                # Get image info
                info = hard_images[img_path]
                true_class = info['true_class']
                
                # Construct title with predictions
                title = f"True: {true_class}\n"
                for version in versions:
                    if version in info['predictions']:
                        pred = info['predictions'][version]['predicted_class']
                        conf = info['predictions'][version]['confidence']
                        title += f"{version}: {pred} ({conf:.2f})\n"
                
                ax.set_title(title, fontsize=8)
                ax.axis('off')
                
                # Also save individual image with annotations
                save_annotated_image(img_path, info, versions)
                
            except Exception as e:
                logger.error(f"Error visualizing image {img_path}: {e}")
                ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('analysis/hard_images/sample_hard_images.png', dpi=300)
    plt.close()

def save_annotated_image(img_path, info, versions):
    """Save an annotated version of the image with prediction information"""
    try:
        # Load image
        img = Image.open(img_path)
        
        # Create a new image with additional space for annotations
        width, height = img.size
        new_height = height + 60 * len(versions)
        annotated_img = Image.new('RGB', (width, new_height), (255, 255, 255))
        annotated_img.paste(img, (0, 0))
        
        # Add annotations
        draw = ImageDraw.Draw(annotated_img)
        try:
            # Try to use a nicer font if available
            font = ImageFont.truetype("Arial", 12)
        except IOError:
            # Fall back to default
            font = ImageFont.load_default()
        
        # Draw true class
        true_class = info['true_class']
        draw.text((10, height + 5), f"True class: {true_class}", fill=(0, 0, 0), font=font)
        
        # Draw predictions for each version
        for i, version in enumerate(versions):
            if version in info['predictions']:
                pred = info['predictions'][version]['predicted_class']
                conf = info['predictions'][version]['confidence']
                text = f"{version} predicted: {pred} (conf: {conf:.2f})"
                draw.text((10, height + 25 + i * 20), text, fill=(200, 0, 0), font=font)
        
        # Save the annotated image
        base_name = os.path.basename(img_path)
        save_path = f"analysis/hard_images/samples/annotated_{base_name}"
        annotated_img.save(save_path)
        
    except Exception as e:
        logger.error(f"Error creating annotated image for {img_path}: {e}")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Find and analyze hard-to-learn images
    find_hard_images()