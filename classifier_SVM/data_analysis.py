import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def load_data(data_dir):
    """Load training, validation and test datasets"""
    train_df = pd.read_csv(os.path.join(data_dir, 'Train_Data_by_AR_png_224.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'Validation_Data_by_AR_png_224.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'Test_Data_by_AR_png_224.csv'))
    return train_df, val_df, test_df

def analyze_folder_distribution(base_dir):
    """Analyze the distribution of images across folders"""
    folder_counts = {}
    total_images = 0
    
    # Get all folders in the base directory
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        # Count number of png files in each folder
        num_images = len([f for f in os.listdir(folder_path) if f.endswith('.png')])
        folder_counts[folder] = num_images
        total_images += num_images
    
    # Calculate statistics
    avg_images = np.mean(list(folder_counts.values()))
    median_images = np.median(list(folder_counts.values()))
    std_images = np.std(list(folder_counts.values()))
    
    # Print statistics
    print("\nFolder Distribution Statistics:")
    print(f"Total number of folders: {len(folder_counts)}")
    print(f"Total number of images: {total_images}")
    print(f"Average images per folder: {avg_images:.2f}")
    print(f"Median images per folder: {median_images:.2f}")
    print(f"Standard deviation: {std_images:.2f}")
    
    # Create histogram of folder sizes
    plt.figure(figsize=(12, 6))
    plt.hist(list(folder_counts.values()), bins=50)
    plt.title('Distribution of Images per Folder')
    plt.xlabel('Number of Images')
    plt.ylabel('Number of Folders')
    plt.tight_layout()
    plt.savefig('/Users/danilgarmaev/Documents/Masters_Research/AR-flares/results/plots/folder_distribution.png')
    plt.close()
    
    # Print top 10 folders with most images
    print("\nTop 10 folders with most images:")
    sorted_folders = sorted(folder_counts.items(), key=lambda x: x[1], reverse=True)
    for folder, count in sorted_folders[:10]:
        print(f"Folder {folder}: {count} images")
    
    return folder_counts

def analyze_class_distribution(train_df, val_df, test_df):
    """Analyze and plot class distribution across datasets"""
    datasets = {
        'Training': train_df,
        'Validation': val_df,
        'Test': test_df
    }
    
    plt.figure(figsize=(10, 6))
    for name, df in datasets.items():
        class_counts = df['class'].value_counts()
        total = len(df)
        print(f"\n{name} Set Class Distribution:")
        print(f"Total samples: {total}")
        print(f"Class 0 (No Flare): {class_counts[0]} ({class_counts[0]/total*100:.2f}%)")
        print(f"Class 1 (Flare): {class_counts[1]} ({class_counts[1]/total*100:.2f}%)")
        
        # Plot
        plt.bar([f"{name}\nClass 0", f"{name}\nClass 1"], 
                [class_counts[0], class_counts[1]], 
                label=name)
    
    plt.title('Class Distribution Across Datasets')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/Users/danilgarmaev/Documents/Masters_Research/AR-flares/results/plots/class_distribution.png')
    plt.close()

def analyze_feature_distributions(df, feature_columns):
    """Analyze and plot distributions of numerical features"""
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(feature_columns, 1):
        plt.subplot(3, 3, i)
        sns.histplot(data=df, x=feature, hue='class', bins=50)
        plt.title(f'{feature} Distribution')
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()

def analyze_correlations(df, feature_columns):
    """Analyze and plot correlations between features"""
    correlation_matrix = df[feature_columns].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('/Users/danilgarmaev/Documents/Masters_Research/AR-flares/results/plots/correlation_matrix.png')
    plt.close()

def analyze_time_distribution(df):
    """Analyze distribution of samples over time"""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='hour', hue='class', bins=24)
    plt.title('Sample Distribution by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    plt.savefig('/Users/danilgarmaev/Documents/Masters_Research/AR-flares/results/plots/time_distribution.png')
    plt.close()

def main():
    # Create output directory if it doesn't exist
    output_dir = Path('/Users/danilgarmaev/Documents/Masters_Research/AR-flares/results/plots')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    data_dir = '/Users/danilgarmaev/Documents/Masters_Research/AR-flares/data/cnn_features'
    train_df, val_df, test_df = load_data(data_dir)
    
    # Analyze folder distribution
    images_dir = '/Users/danilgarmaev/Documents/Masters_Research/AR-flares/data/Lat60_Lon60_Nans0_png_224'
    folder_counts = analyze_folder_distribution(images_dir)
    
    # Analyze class distribution
    analyze_class_distribution(train_df, val_df, test_df)
    
    # Get numerical feature columns (excluding 'class' and 'timestamp')
    feature_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in feature_columns if col not in ['class', 'timestamp']]
    
    # Analyze feature distributions
    # analyze_feature_distributions(train_df, feature_columns)
    
    # Analyze correlations
    # analyze_correlations(train_df, feature_columns)
    
    # # Analyze time distribution
    # analyze_time_distribution(train_df)
    
    # Print summary statistics
    # print("\nSummary Statistics for Training Set:")
    # print(train_df[feature_columns].describe())
    
    # Calculate and print class imbalance ratio
    class_ratio = len(train_df[train_df['class'] == 0]) / len(train_df[train_df['class'] == 1])
    print(f"\nClass Imbalance Ratio (No Flare:Flare): {class_ratio:.2f}")

if __name__ == "__main__":
    main() 