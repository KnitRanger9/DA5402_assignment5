import os
import yaml
import logging
import pickle
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import itertools

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/train_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def build_model(config, input_shape=(32, 32, 3), num_classes=10):
    """Build a CNN model based on configuration parameters"""
    model = models.Sequential()
    
    # Add convolutional layers
    for i in range(config['model']['conv_layers']):
        filters = config['model']['conv_filters'][i]
        kernel_size = config['model']['kernel_sizes'][i]
        
        # First layer needs input_shape
        if i == 0:
            model.add(layers.Conv2D(filters, kernel_size, activation='relu', padding='same', input_shape=input_shape))
        else:
            model.add(layers.Conv2D(filters, kernel_size, activation='relu', padding='same'))
        
        # Add max pooling after each conv layer
        model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and add dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(config['model']['dropout_rate']))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config['model']['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_tune_model():
    """Train and tune the model with hyperparameter search"""
    # Load configuration
    config = load_config()
    random_seed = config['random_seed']
    
    # Set random seed
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    # Load datasets
    with open('data/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    with open('data/val_data.pkl', 'rb') as f:
        val_data = pickle.load(f)
    
    # Extract data
    X_train = train_data['images']
    y_train = train_data['labels']
    X_val = val_data['images']
    y_val = val_data['labels']
    class_names = train_data['class_names']
    num_classes = len(class_names)
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")
    logger.info(f"Number of classes: {num_classes}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    # Get hyperparameters for tuning
    param1 = config['tuning']['param1']
    param1_values = config['tuning']['param1_values']
    param2 = config['tuning']['param2']
    param2_values = config['tuning']['param2_values']
    
    logger.info(f"Hyperparameter tuning: {param1}={param1_values}, {param2}={param2_values}")
    
    # Prepare for hyperparameter tuning
    best_val_accuracy = 0
    best_model = None
    best_params = {}
    tuning_results = []
    
    # Create hyperparameter combinations
    param_combinations = list(itertools.product(param1_values, param2_values))
    
    # Loop through hyperparameter combinations
    for i, (p1_value, p2_value) in enumerate(param_combinations):
        logger.info(f"Training model {i+1}/{len(param_combinations)}: {param1}={p1_value}, {param2}={p2_value}")
        
        # Create a copy of the config and update with current hyperparameters
        current_config = load_config()
        
        # Update hyperparameters
        if param1 == 'learning_rate':
            current_config['model']['learning_rate'] = p1_value
        elif param1 == 'dropout_rate':
            current_config['model']['dropout_rate'] = p1_value
        elif param1.startswith('conv_'):
            # Handle conv_layers, conv_filters, etc.
            parts = param1.split('.')
            if len(parts) == 1:
                current_config['model'][param1] = p1_value
            else:
                # Handle nested parameters like conv_filters[0]
                param_name = parts[0]
                index = int(parts[1])
                current_config['model'][param_name][index] = p1_value
        
        if param2 == 'learning_rate':
            current_config['model']['learning_rate'] = p2_value
        elif param2 == 'dropout_rate':
            current_config['model']['dropout_rate'] = p2_value
        elif param2.startswith('conv_'):
            # Handle conv_layers, conv_filters, etc.
            parts = param2.split('.')
            if len(parts) == 1:
                current_config['model'][param2] = p2_value
            else:
                # Handle nested parameters like conv_filters[0]
                param_name = parts[0]
                index = int(parts[1])
                current_config['model'][param_name][index] = p2_value
        
        # Build model with current hyperparameters
        model = build_model(current_config, input_shape=X_train.shape[1:], num_classes=num_classes)
        
        # Callbacks
        checkpoint_path = f'models/model_{i+1}.h5'
        checkpoint = ModelCheckpoint(
            checkpoint_path, 
            save_best_only=True, 
            monitor='val_accuracy', 
            mode='max'
        )
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=current_config['model']['batch_size'],
            epochs=current_config['model']['epochs'],
            validation_data=(X_val, y_val),
            callbacks=[checkpoint, early_stopping],
            verbose=1
        )
        
        # Evaluate model
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"Validation accuracy: {val_accuracy:.4f}, Loss: {val_loss:.4f}")
        
        # Save training history
        training_history = {
            'accuracy': [float(acc) for acc in history.history['accuracy']],
            'val_accuracy': [float(acc) for acc in history.history['val_accuracy']],
            'loss': [float(loss) for loss in history.history['loss']],
            'val_loss': [float(loss) for loss in history.history['val_loss']]
        }
        
        # Record results
        result = {
            'model_id': i+1,
            'hyperparameters': {
                param1: p1_value,
                param2: p2_value
            },
            'val_accuracy': float(val_accuracy),
            'val_loss': float(val_loss),
            'epochs_trained': len(history.history['accuracy']),
            'training_history': training_history
        }
        tuning_results.append(result)
        
        # Check if this is the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model
            best_params = {
                param1: p1_value,
                param2: p2_value
            }
            logger.info(f"New best model found with validation accuracy: {best_val_accuracy:.4f}")
    
    # Save the best model
    best_model.save('models/best_model.h5')
    logger.info(f"Best model saved with parameters: {best_params}")
    
    # Save tuning results
    with open('models/tuning_results.json', 'w') as f:
        json.dump(tuning_results, f, indent=2)
    
    # Save training metrics
    training_metrics = {
        'best_val_accuracy': float(best_val_accuracy),
        'best_hyperparameters': best_params,
        'num_models_trained': len(tuning_results)
    }
    
    with open('metrics/training_metrics.json', 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    logger.info("Model training and tuning completed successfully")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Train and tune model
    train_and_tune_model()