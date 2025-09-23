import os
from pathlib import Path
import imageio.v3 as iio
from tqdm import tqdm

def convert_folder_to_tiff(folder_path):
    """Convert all PNG files in a folder to a single TIFF file"""
    folder = Path(folder_path)
    if not folder.exists():
        return
    
    # Get all PNG files and sort them by filename
    png_files = sorted(folder.glob("*.png"))
    if not png_files:
        return
    
    # Create output directory if it doesn't exist
    output_dir = folder.parent / "tiff_files"
    output_dir.mkdir(exist_ok=True)
    
    # Create output filename based on folder name
    output_file = output_dir / f"{folder.name}.tiff"
    
    # Read all PNG files and combine them into a single TIFF
    images = []
    for png_file in png_files:
        try:
            img = iio.imread(png_file)
            images.append(img)
        except Exception as e:
            print(f"Error reading {png_file}: {e}")
    
    if images:
        try:
            # Save as multi-page TIFF
            iio.imwrite(output_file, images)
            print(f"Successfully created {output_file}")
        except Exception as e:
            print(f"Error saving {output_file}: {e}")

def main():
    base_dir = Path("data/Lat60_Lon60_Nans0_png_224")
    folders = list(base_dir.glob("*"))
    
    print("Converting folders to TIFF:")
    for folder in tqdm(folders):
        if folder.is_dir():
            convert_folder_to_tiff(folder)

if __name__ == "__main__":
    main()
