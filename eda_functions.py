import os
from PIL.ExifTags import TAGS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
import cv2


def extract_metadata(directory):
    """
    Extracts metadata from images in a given directory, including dimensions,
    file size, EXIF data,
    and GPS coordinates if available.

    Args:
        directory (str): Path to the directory containing images.

    Returns:
        pd.DataFrame: A DataFrame containing extracted metadata for each image.
    """
    metadata_list = []

    for image_path in tqdm(Path(directory).rglob("*.jpeg"),
                           desc=f"Extracting metadata from {directory}"):
        try:
            img = Image.open(image_path)
            img_exif = img._getexif()

            img_metadata = {
                "file_name": image_path.name,
                "file_path": str(image_path),
                "width": img.width,
                "height": img.height,
                "aspect_ratio": round(img.width / img.height, 2),
                "image_size_kb": round(os.path.getsize(image_path) / 1024, 2),
                "has_exif": bool(img_exif),
                "capture_date": None,
                "latitude": None,
                "longitude": None
            }

            if img_exif:
                exif_data = {TAGS.get(tag, tag): val for tag,
                             val in img_exif.items()}
                img_metadata["capture_date"] = exif_data.get(
                    "DateTimeOriginal")

                gps_info = exif_data.get("GPSInfo")
                if gps_info:
                    lat_ref = gps_info.get(1)
                    lon_ref = gps_info.get(3)
                    lat = gps_info.get(2)
                    lon = gps_info.get(4)

                    if lat and lon:
                        img_metadata["latitude"] = \
                            (lat[0] + lat[1]/60 + lat[2]/3600) * \
                            (-1 if lat_ref == "S" else 1)
                        img_metadata["longitude"] = \
                            (lon[0] + lon[1]/60 + lon[2]/3600) * \
                            (-1 if lon_ref == "W" else 1)

            metadata_list.append(img_metadata)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return pd.DataFrame(metadata_list)


def plot_image_size_distribution(df_metadata):
    """
    Plots the distribution of image file sizes in kilobytes (KB).

    Args:
        df_metadata (pd.DataFrame): DataFrame containing image metadata,
        including file sizes.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(df_metadata["image_size_kb"], bins=50, kde=True, color="blue",
                 alpha=0.7)
    plt.xlabel("Image Size (KB)")
    plt.ylabel("Number of Images")
    plt.title("Distribution of Image File Sizes")
    plt.show()


def show_extreme_size_images(df_metadata, num_samples=3):
    """
    Displays a selection of the smallest and largest images by file size.

    Args:
        df_metadata (pd.DataFrame): DataFrame containing image metadata,
        including file sizes.
        num_samples (int, optional): Number of smallest and largest images to
        display. Defaults to 3.

    Returns:
        None
    """
    smallest = df_metadata.nsmallest(num_samples, "image_size_kb")
    largest = df_metadata.nlargest(num_samples, "image_size_kb")

    for category, images in [("Smallest", smallest), ("Largest", largest)]:
        plt.figure(figsize=(12, 4))
        plt.suptitle(f"{category} File Size Images", fontsize=14)
        for i, img_path in enumerate(images["file_path"]):
            img = Image.open(img_path)
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(img)
            plt.axis("off")
        plt.show()


def show_random_small_images(small_images, num_samples=15):
    """
    Displays a random selection of the smallest images based on file size.

    Args:
        small_images (pd.DataFrame): DataFrame containing metadata for small
        images.
        num_samples (int, optional): Number of images to display. Defaults to
        15.

    Returns:
        None
    """
    sampled_images = random.sample(list(small_images["file_path"]),
                                   min(num_samples, num_small_images))

    plt.figure(figsize=(12, 6))
    plt.suptitle("Random 15 Smallest Images", fontsize=14)
    for i, img_path in enumerate(sampled_images):
        img = Image.open(img_path)
        plt.subplot(3, 5, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()


def remove_small_images(df_metadata, size_threshold_kb=1.5):
    """
    Deletes images below a specified file size threshold.

    Args:
        df_metadata (pd.DataFrame): DataFrame containing image metadata,
        including file sizes.
        size_threshold_kb (float, optional): The file size threshold in KB.
        Defaults to 1.5 KB.

    Returns:
        None
    """
    small_files = df_metadata[df_metadata["image_size_kb"] <
                              size_threshold_kb]["file_path"]

    deleted = 0
    for file_path in small_files:
        try:
            os.remove(file_path)
            deleted += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    print(f"✅ Deleted {deleted} images below {size_threshold_kb} KB.")


def compute_rgb_distributions(directory):
    """
    Computes the average red (R), green (G), and blue (B) channel values for a
    sample of images.

    Args:
        directory (str): Path to the directory containing images.

    Returns:
        tuple: Three NumPy arrays containing mean intensity values for the R,
        G, and B channels.
    """
    r_vals, g_vals, b_vals = [], [], []

    image_paths = list(Path(directory).rglob("*.jpeg"))
    sampled_images = np.random.choice(image_paths, min(500, len(image_paths)),
                                      replace=False)

    for image_path in tqdm(sampled_images,
                           desc=f"Analyzing RGB channels in {directory}"):
        try:
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

            r_vals.append(np.mean(img[:, :, 0]))  # Red
            g_vals.append(np.mean(img[:, :, 1]))  # Green
            b_vals.append(np.mean(img[:, :, 2]))  # Blue

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return np.array(r_vals), np.array(g_vals), np.array(b_vals)


def plot_rgb_histograms(r_vals, g_vals, b_vals):
    """
    Plots histograms for the distributions of red, green, and blue channel
    intensities.

    Args:
        r_vals (np.array): Array of mean red channel values.
        g_vals (np.array): Array of mean green channel values.
        b_vals (np.array): Array of mean blue channel values.

    Returns:
        None
    """
    plt.figure(figsize=(12, 5))
    plt.hist(r_vals, bins=50, color="red", alpha=0.6, label="Red Channel")
    plt.hist(g_vals, bins=50, color="green", alpha=0.6, label="Green Channel")
    plt.hist(b_vals, bins=50, color="blue", alpha=0.6, label="Blue Channel")
    plt.xlabel("Mean Intensity")
    plt.ylabel("Number of Images")
    plt.title("RGB Channel Distributions")
    plt.legend()
    plt.show()


def show_most_blue_images(directory, num_samples=5):
    """
    Identifies and displays images with the highest dominance of the blue
    channel.

    Args:
        directory (str): Path to the directory containing images.
        num_samples (int, optional): Number of images to display. Defaults to
        5.

    Returns:
        None
    """
    blue_ratios = []

    for image_path in Path(directory).rglob("*.jpeg"):
        try:
            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            r_mean, g_mean, b_mean = np.mean(img[:, :, 0]),
            np.mean(img[:, :, 1]), np.mean(img[:, :, 2])
            blue_ratio = b_mean / (r_mean + g_mean + b_mean)

            blue_ratios.append((image_path, blue_ratio))

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    blue_ratios.sort(key=lambda x: x[1], reverse=True)
    sampled_images = [img_path for img_path, _ in blue_ratios[:num_samples]]

    # Display the images
    plt.figure(figsize=(12, 4))
    plt.suptitle("Most Blue-Dominant Images", fontsize=14)
    for i, img_path in enumerate(sampled_images):
        img = Image.open(img_path)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()


def compare_rgb_distributions(directory):
    """
    Compares RGB distributions for images in 'damage' and 'no_damage'
    subdirectories.

    Args:
        directory (str): Path to the main directory containing 'damage' and
        'no_damage' subdirectories.

    Returns:
        None
    """
    damage_dir = Path(directory) / "damage"
    no_damage_dir = Path(directory) / "no_damage"

    r_damage, g_damage, b_damage = compute_rgb_distributions(damage_dir)
    r_no_damage, g_no_damage, b_no_damage = compute_rgb_distributions(
        no_damage_dir)

    plt.figure(figsize=(12, 5))
    plt.hist(r_damage, bins=50, color="red", alpha=0.5, label="Damage - Red")
    plt.hist(r_no_damage, bins=50, color="red", alpha=0.2, linestyle="dashed",
             label="No Damage - Red")

    plt.hist(g_damage, bins=50, color="green", alpha=0.5,
             label="Damage - Green")
    plt.hist(g_no_damage, bins=50, color="green", alpha=0.2,
             linestyle="dashed", label="No Damage - Green")

    plt.hist(b_damage, bins=50, color="blue", alpha=0.5, label="Damage - Blue")
    plt.hist(b_no_damage, bins=50, color="blue", alpha=0.2, linestyle="dashed",
             label="No Damage - Blue")

    plt.xlabel("Mean Intensity")
    plt.ylabel("Number of Images")
    plt.title("RGB Distribution: Damaged vs. Non-Damaged Images")
    plt.legend()
    plt.show()


def plot_separate_rgb_distributions(r_damage, g_damage, b_damage, r_no_damage,
                                    g_no_damage, b_no_damage):
    """
    Plots separate histograms for the red, green, and blue channel
    distributions
    for damaged and non-damaged images.

    Args:
        r_damage (np.array): Red channel values for damaged images.
        g_damage (np.array): Green channel values for damaged images.
        b_damage (np.array): Blue channel values for damaged images.
        r_no_damage (np.array): Red channel values for non-damaged images.
        g_no_damage (np.array): Green channel values for non-damaged images.
        b_no_damage (np.array): Blue channel values for non-damaged images.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(r_damage, bins=50, color="red", alpha=0.5,
                 label="Damage - Red")
    axes[0].hist(r_no_damage, bins=50, color="pink", alpha=0.5,
                 label="No Damage - Red")
    axes[0].set_title("Red Channel Distribution")
    axes[0].set_xlabel("Mean Intensity")
    axes[0].set_ylabel("Number of Images")
    axes[0].legend()

    axes[1].hist(g_damage, bins=50, color="green", alpha=0.5,
                 label="Damage - Green")
    axes[1].hist(g_no_damage, bins=50, color="lightgreen", alpha=0.5,
                 label="No Damage - Green")
    axes[1].set_title("Green Channel Distribution")
    axes[1].set_xlabel("Mean Intensity")
    axes[1].legend()

    axes[2].hist(b_damage, bins=50, color="blue", alpha=0.5,
                 label="Damage - Blue")
    axes[2].hist(b_no_damage, bins=50, color="lightblue", alpha=0.5,
                 label="No Damage - Blue")
    axes[2].set_title("Blue Channel Distribution")
    axes[2].set_xlabel("Mean Intensity")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def compute_rgb_means(df_metadata):
    """
    Computes and stores the average red (R), green (G), and blue (B) channel
    values for each image
    in the provided metadata DataFrame.

    Args:
        df_metadata (pd.DataFrame): DataFrame containing image file paths.

    Returns:
        pd.DataFrame: Updated DataFrame with new columns for average R, G, and
        B values.
    """
    r_vals, g_vals, b_vals = [], [], []

    for image_path in tqdm(df_metadata["file_path"],
                           desc="Computing RGB means"):
        try:
            img = cv2.imread(str(image_path))

            if img is None:
                print(f"⚠️ Skipping unreadable file: {image_path}")
                r_vals.append(None)
                g_vals.append(None)
                b_vals.append(None)
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

            r_vals.append(np.mean(img[:, :, 0]))  # Red channel mean
            g_vals.append(np.mean(img[:, :, 1]))  # Green channel mean
            b_vals.append(np.mean(img[:, :, 2]))  # Blue channel mean

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            r_vals.append(None)
            g_vals.append(None)
            b_vals.append(None)

    df_metadata["r_mean"] = r_vals
    df_metadata["g_mean"] = g_vals
    df_metadata["b_mean"] = b_vals

    return df_metadata


def show_extreme_color_images(df_metadata, color, threshold=50, num_samples=5):
    """
    Displays images with the lowest intensity in a specified color channel.

    Args:
        df_metadata (pd.DataFrame): DataFrame containing image metadata and
        RGB mean values.
        color (str): The color channel to filter by ('red', 'green', 'blue').
        threshold (int, optional): The intensity threshold for selecting
        images. Defaults to 50.
        num_samples (int, optional): Number of images to display. Defaults to
        5.

    Returns:
        None
    """
    if color == "red":
        selected_images = df_metadata[df_metadata["r_mean"] < threshold]
        title = "Low-Red Images (Possible Burnt/Damaged Areas)"
    elif color == "green":
        selected_images = df_metadata[df_metadata["g_mean"] < threshold]
        title = "Low-Green Images (Possible Vegetation Loss)"
    elif color == "blue":
        selected_images = df_metadata[df_metadata["b_mean"] < threshold]
        title = "Low-Blue Images (Possible Flooding or Shadows)"

    sampled_images = random.sample(list(selected_images["file_path"]),
                                   min(num_samples, len(selected_images)))

    plt.figure(figsize=(12, 4))
    plt.suptitle(title, fontsize=14)
    for i, img_path in enumerate(sampled_images):
        img = Image.open(img_path)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()
