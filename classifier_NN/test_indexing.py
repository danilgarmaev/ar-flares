import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import tifffile
import glob

# Environment-specific paths
IS_COMPUTE_CANADA = os.path.exists('/scratch')
BASE_PATH = '/scratch/dgarmaev/AR-flares/data' if IS_COMPUTE_CANADA else '/Users/danilgarmaev/Documents/Masters_Research/AR-flares/data'
TIFF_PATH = os.path.join(BASE_PATH, 'tiff_files')
LABELS_PATH = os.path.join(BASE_PATH, 'cnn_features')


class ARFlaresDataset(Dataset):
    def __init__(self, tiff_files, labels_map, transform=None):
        self.transform = transform
        self.labels_map = labels_map
        self.image_index = []
        self.tiff_file_index = {}

        print("\n🌀 Indexing TIFF files...")
        for tiff_path in tiff_files:
            tiff_name = os.path.basename(tiff_path)
            tiff_number = tiff_name.split('.')[0]

            with tifffile.TiffFile(tiff_path) as tif:
                num_images = len(tif.pages)
            
            if tiff_number not in labels_map:
                print(f"⚠️ Warning: No label found for TIFF {tiff_number}, skipping.")
                continue

            self.tiff_file_index[tiff_number] = tiff_path
            for img_idx in range(num_images):
                self.image_index.append((tiff_number, img_idx))

        print(f"✅ Indexed {len(self.image_index)} images from {len(self.tiff_file_index)} TIFFs")

    def __len__(self):
        return len(self.image_index)

    def __getitem__(self, idx):
        tiff_number, img_idx = self.image_index[idx]
        tiff_path = self.tiff_file_index[tiff_number]

        with tifffile.TiffFile(tiff_path) as tif:
            image = tif.pages[img_idx].asarray()

        image = Image.fromarray(image).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = int(self.labels_map[tiff_number])
        return image, label

    @staticmethod
    def create_labels_map(csv_path):
        df = pd.read_csv(csv_path, dtype=str)
        labels_map = {}
        for _, row in df.iterrows():
            tiff_number = row['filename'].split('/')[0]
            labels_map[tiff_number] = row['class']
        return labels_map


def get_tiff_files(limit=None):
    all_files = sorted(glob.glob(os.path.join(TIFF_PATH, '*.tiff')))
    return all_files if limit is None else all_files[:limit]


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


BATCH_SIZE = 32
NUM_WORKERS = 2

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Optional normalization
    # transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def create_dataset(csv_filename, max_files=None):
    csv_path = os.path.join(LABELS_PATH, csv_filename)
    labels_map = ARFlaresDataset.create_labels_map(csv_path)

    tiff_files = get_tiff_files()
    if max_files:
        tiff_files = tiff_files[:max_files]
    
    # Filter TIFF files to only those with a label
    tiff_files = [f for f in tiff_files if os.path.basename(f).split('.')[0] in labels_map]

    dataset = ARFlaresDataset(tiff_files, labels_map, transform=transform)
    return dataset

def preview_dataset(name, csv_file, max_files=None):
    print(f"\n🚀 Creating {name} dataset...")
    dataset = create_dataset(csv_file, max_files=max_files)
    
    print(f"\n🔍 Previewing {name} samples...")
    for i in range(min(5, len(dataset))):
        img, label = dataset[i]
        print(f"{name} Sample {i}: Image shape = {img.shape}, Label = {label}")
    return dataset

def create_dataloaders(datasets_dict):
    dataloaders = {}
    for name, dataset in datasets_dict.items():
        shuffle = (name == "Train")
        dataloaders[name] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
    return dataloaders

def main():
    datasets_csvs = {
        "Train": "Train_Data_by_AR_png_224.csv",
        "Validation": "Validation_Data_by_AR_png_224.csv",
        "Test": "Test_Data_by_AR_png_224.csv"
    }

    all_datasets = {}
    for name, csv_file in datasets_csvs.items():
        dataset = preview_dataset(name, csv_file, max_files=None)  # Set to None for full set
        all_datasets[name] = dataset

    print("\n📊 Dataset Summary:")
    for name, dataset in all_datasets.items():
        print(f"{name}: {len(dataset)} images")

    dataloaders = create_dataloaders(all_datasets)

    return all_datasets, dataloaders


if __name__ == '__main__':
    all_datasets, dataloaders = main()