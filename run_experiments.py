import os
import yaml
import logging
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shutil
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/experiments.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def update_config(version, seed):
    """Update configuration file with specified dataset version and random seed"""
    # Load existing config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update version and seed
    config['dataset']['version'] = version
    config['random_seed'] = seed
    
    # Save updated config
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Updated config: version={version}, seed={seed}")

def create_combined_datasets():
    """Create combined dataset versions v1+v2 and v1+v2+v3"""
    try:
        # Pull all individual partitions
        subprocess.run(['dvc', 'checkout', 'partitions/partition_v0/data.dvc'], check=True)
        subprocess.run(['dvc', 'checkout', 'partitions/partition_v1/data.dvc'], check=True)
        subprocess.run(['dvc', 'checkout', 'partitions/partition_v2/data.dvc'], check=True)
        
        # Create v0+v1 combined dataset
        v0v1_dir = 'partitions/partition_v0v1'
        
        os.makedirs(v0v1_dir, exist_ok = True)
        
        # Copy class directories and images
        class_names = os.listdir('partitions/partition_v0/data')
        for class_name in class_names:
            class_dir = os.path.join(v0v1_dir, class_name)
            os.makedirs(class_dir, exist_ok = True)
            
            # Copy images from v0

            for file_path in glob.glob('partitions/partition_v0/data/{class_name}/*'):
                if os.path.isfile(file_path):
                    try:
                        shutil.copy(file_path, class_dir)
                    except Exception as e:
                        print(e)
            
            # Copy images from v1
            for file_path in glob.glob('partitions/partition_v1/data/{class_name}/*'):
                if os.path.isfile(file_path):
                    try:
                        shutil.copy(file_path, class_dir)
                    except Exception as e:
                        print(e)
        
        # Create v0+v1+v2 combined dataset
        v0v1v2_dir = 'partitions/partition_v0v1v2'
        
        os.makedirs(v0v1v2_dir, exist_ok = True)
        
        # Copy class directories and images
        for class_name in class_names:
            class_dir = os.path.join(v0v1v2_dir, class_name)
            os.makedirs(class_dir, exist_ok = True)
            
            # Copy images from v0
            for file_path in glob.glob('partitions/partition_v0/data/{class_name}/*'):
                if os.path.isfile(file_path):
                    try:
                        shutil.copy(file_path, class_dir)
                    except Exception as e:
                        print(e)
            
            # Copy images from v1
            for file_path in glob.glob('partitions/partition_v1/data/{class_name}/*'):
                if os.path.isfile(file_path):
                    try:
                        shutil.copy(file_path, class_dir)
                    except Exception as e:
                        print(e)
            
            # Copy images from v2
            for file_path in glob.glob('partitions/partition_v2/data/{class_name}/*'):
                if os.path.isfile(file_path):
                    try:
                        shutil.copy(file_path, class_dir)
                    except Exception as e:
                        print(e)
        
        # Add combined datasets to DVC
        subprocess.run(['dvc', 'add', f"{v0v1_dir}/data"], check=True, shell = True)
        subprocess.run(['dvc', 'add', f"{v0v1v2_dir}/data"], check=True, shell = True)
        
        # Add to Git
        subprocess.run(['git', 'add', f'{v0v1_dir}/data.dvc {v0v1_dir}/.gitignore'], check=True, shell = True)
        subprocess.run(['git', 'add', f'{v0v1v2_dir}/data.dvc {v0v1v2_dir}/.gitignore'], check=True, shell = True)
        
        # Commit changes
        subprocess.run(['git', 'commit', '-m', 'Add combined dataset partitions'], check=True, shell = True)
        
        logger.info("Created and added combined dataset partitions to DVC")
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating combined datasets: {e}")
        raise

def run_experiment(version, seed):
    """Run a complete experiment with the specified dataset version and random seed"""
    # Update config file
    update_config(version, seed)
    
    # Create experiment name
    exp_name = f"{version}_seed{seed}"
    
    # Run DVC experiment
    try:
        logger.info(f"Starting experiment: {exp_name}")
        
        # Run DVC experiment
        result = subprocess.run(
            ['dvc', 'exp', 'run', '--name', exp_name],
            check=False,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"Experiment {exp_name} failed with exit code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            logger.error(f"stdout: {result.stdout}")
        
        logger.info(f"Experiment {exp_name} completed")
        logger.info(result.stdout)
        
        # Return the experiment name for reference
        return exp_name
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running experiment {exp_name}: {e}")
        logger.error(e.stderr)
        raise

def run_all_experiments():
    """Run all required experiments with different dataset versions and seeds"""
    # Create combined datasets
    create_combined_datasets()
    
    # Versions to test
    versions = ['v0', 'v1', 'v2', 'v0v1', 'v0v1v2']
    # Seeds to test
    seeds = [42, 99, 123]
    
    # Track all experiments
    experiments = []
    
    # Run experiments for each version and seed
    for version in versions:
        for seed in seeds:
            exp_name = run_experiment(version, seed)
            experiments.append({
                'version': version,
                'seed': seed,
                'experiment': exp_name
            })
    
    logger.info(f"All experiments completed: {len(experiments)}")
    
    # Show DVC experiment comparison
    try:
        result = subprocess.run(
            ['dvc', 'exp', 'show'],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("DVC Experiment Comparison:")
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error showing experiments: {e}")
    
    # Analyze results
    analyze_results(experiments)

def analyze_results(experiments):
    """Analyze and visualize results from all experiments"""
    # Create directory for analysis results
    os.makedirs('analysis', exist_ok=True)
    
    # Collect metrics from all experiments
    results = []
    
    for exp in experiments:
        try:
            # Load evaluation metrics for this experiment
            metrics_file = f"metrics/evaluation_metrics.json"
            
            # Check if file exists
            if not os.path.exists(metrics_file):
                logger.warning(f"Metrics file not found for {exp['experiment']}")
                continue
            
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Add experiment metadata
            result = {
                'version': exp['version'],
                'seed': exp['seed'],
                'experiment': exp['experiment'],
                'accuracy': metrics['overall_accuracy']
            }
            
            # Add class accuracies
            for class_name, accuracy in metrics['class_accuracies'].items():
                result[f'accuracy_{class_name}'] = accuracy
            
            results.append(result)
        
        except Exception as e:
            logger.error(f"Error loading metrics for {exp['experiment']}: {e}")
    
    # Convert to DataFrame for analysis
    if results:
        df = pd.DataFrame(results)
        
        # Save results to CSV
        df.to_csv('analysis/experiment_results.csv', index=False)
        
        # Group by version and calculate statistics
        version_stats = df.groupby('version').agg({
            'accuracy': ['mean', 'std', 'min', 'max']
        })
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        # Plot accuracy by version and seed
        plt.subplot(2, 1, 1)
        for seed in df['seed'].unique():
            seed_data = df[df['seed'] == seed]
            plt.plot(seed_data['version'], seed_data['accuracy'], marker='o', label=f'Seed {seed}')
        
        plt.title('Model Accuracy by Dataset Version and Seed')
        plt.xlabel('Dataset Version')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        
        # Plot mean accuracy with error bars
        plt.subplot(2, 1, 2)
        vers = version_stats.index
        means = version_stats['accuracy']['mean']
        stds = version_stats['accuracy']['std']
        
        plt.bar(vers, means, yerr=stds, capsize=5, alpha=0.7)
        plt.title('Mean Accuracy by Dataset Version (with Standard Deviation)')
        plt.xlabel('Dataset Version')
        plt.ylabel('Mean Accuracy')
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig('analysis/accuracy_comparison.png', dpi=300)
        
        # Print summary statistics
        logger.info("Experiment Results Summary:")
        logger.info(f"\n{version_stats}")
        
        # Save summary to file
        with open('analysis/summary.txt', 'w') as f:
            f.write("Experiment Results Summary:\n\n")
            f.write(str(version_stats))
        
        logger.info("Results analysis completed and saved to analysis directory")
    else:
        logger.warning("No valid results found for analysis")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run all experiments
    run_all_experiments()