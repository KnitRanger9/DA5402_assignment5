import os
import yaml
import logging
import pickle
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/evaluate_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def evaluate_model():
    """Evaluate model performance on test data and generate reports"""
    # Load configuration
    config = load_config()
    random_seed = config['random_seed']
    dataset_version = config['dataset']['version']
    
    # Set random seed
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    # Load test data
    with open('data/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    # Extract data
    X_test = test_data['images']
    y_test = test_data['labels']
    class_names = test_data['class_names']
    num_classes = len(class_names)
    
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info(f"Number of classes: {num_classes}")
    
    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    # Load the best model
    logger.info("Loading the best model")
    model = tf.keras.models.load_model('models/best_model.h5')
    
    # Get model predictions
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calculate overall accuracy
    accuracy = np.mean(y_pred == y_test)
    logger.info(f"Test accuracy: {accuracy:.4f}")
    
    # Calculate class-wise accuracy
    class_accuracies = {}
    for class_idx in range(num_classes):
        class_mask = (y_test == class_idx)
        if np.sum(class_mask) > 0:  # Check if we have samples of this class
            class_acc = np.mean(y_pred[class_mask] == y_test[class_mask])
            class_accuracies[class_names[class_idx]] = float(class_acc)
            logger.info(f"Class {class_names[class_idx]} accuracy: {class_acc:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {dataset_version}')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save confusion matrix
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    logger.info("Confusion matrix saved to results/confusion_matrix.png")
    
    # Get detailed classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    logger.info("Classification report generated")
    
    # Save per-image predictions for later analysis
    image_predictions = []
    for i in range(len(X_test)):
        image_predictions.append({
            'image_path': test_data['paths'][i],
            'true_label': int(y_test[i]),
            'true_class': class_names[y_test[i]],
            'predicted_label': int(y_pred[i]),
            'predicted_class': class_names[y_pred[i]],
            'correct': bool(y_pred[i] == y_test[i]),
            'confidence': float(y_pred_prob[i][y_pred[i]])
        })
    
    # Save all evaluation metrics
    evaluation_metrics = {
        'dataset_version': dataset_version,
        'random_seed': random_seed,
        'overall_accuracy': float(accuracy),
        'class_accuracies': class_accuracies,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'num_test_samples': len(y_test)
    }
    
    with open('metrics/evaluation_metrics.json', 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    
    # Save image predictions for further analysis
    with open('results/image_predictions.json', 'w') as f:
        json.dump(image_predictions, f, indent=2)
    
    logger.info("Model evaluation completed successfully")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Evaluate model
    evaluate_model()