import hashlib
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict


def count_images(directory):
    """
    Counts the number of images in each subdirectory within a given directory.

    Args:
        directory (str): Path to the main directory containing image
        subdirectories.

    Returns:
        dict: A dictionary where keys are subdirectory names and values are
        the number of images in each.
    """
    class_counts = {}
    for class_folder in os.listdir(directory):
        class_path = os.path.join(directory, class_folder)
        if os.path.isdir(class_path):
            class_counts[class_folder] = len(os.listdir(class_path))
    return class_counts


def find_duplicates(directory):
    """
    Identifies duplicate images in a directory based on their MD5 hash.

    Args:
        directory (str): Path to the directory containing images.

    Returns:
        list: A list of duplicate image file paths.
    """
    hashes = defaultdict(list)
    duplicates = []

    for image_path in tqdm(Path(directory).rglob("*.jpeg"),
                           desc=f"Checking duplicates in {directory}"):
        try:
            with open(image_path, "rb") as f:
                img_hash = hashlib.md5(f.read()).hexdigest()
            hashes[img_hash].append(image_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    for hash_value, paths in hashes.items():
        if len(paths) > 1:
            duplicates.extend(paths[1:])
    return duplicates


def check_image_sizes(directory):
    """
    Checks the sizes of images within a directory and counts occurrences of
    each unique size.

    Args:
        directory (str): Path to the directory containing images.

    Returns:
        dict: A dictionary where keys are image sizes (width, height) tuples,
        and values are the count of images with that size.
    """
    sizes = defaultdict(int)

    for image_path in tqdm(Path(directory).rglob("*.jpeg"),
                           desc=f"Checking image sizes in {directory}"):
        try:
            img = Image.open(image_path)
            sizes[img.size] += 1
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return sizes


def is_blurry(image_path, threshold=100):
    """
    Determines if an image is blurry using the Laplacian variance method.

    Args:
        image_path (str or Path): Path to the image file.
        threshold (float, optional): Blurriness threshold. Lower variance
        values indicate blurriness. Defaults to 100.

    Returns:
        bool: True if the image is blurry, False otherwise.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return True
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    return variance < threshold


def find_blurry_images(directory):
    """
    Identifies blurry images within a directory.

    Args:
        directory (str): Path to the directory containing images.

    Returns:
        list: A list of paths to blurry images.
    """
    blurry_images = []

    for image_path in tqdm(Path(directory).rglob("*.jpeg"),
                           desc=f"Checking blurriness in {directory}"):
        if is_blurry(image_path):
            blurry_images.append(str(image_path))

    return blurry_images


def plot_random_images(directory, num_images=5):
    """
    Randomly selects and displays a set number of images from each class
    (subdirectory).

    Args:
        directory (str): Path to the directory containing subdirectories with
        images.
        num_images (int, optional): Number of images to display per class.
        Defaults to 5.

    Returns:
        None
    """
    class_dirs = [d for d in Path(directory).iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        images = list(class_dir.glob("*.jpeg"))
        sampled_images = np.random.choice(images, min(num_images, len(images)),
                                          replace=False)

        plt.figure(figsize=(12, 4))
        plt.suptitle(f"Class: {class_dir.name}", fontsize=14)
        for i, img_path in enumerate(sampled_images):
            img = Image.open(img_path)
            plt.subplot(1, num_images, i + 1)
            plt.imshow(img)
            plt.axis("off")
        plt.show()


def compute_blurriness(image_path):
    """
    Computes the blurriness of an image using the Laplacian variance.

    Args:
        image_path (str or Path): Path to the image file.

    Returns:
        float or None: Laplacian variance value, where lower values indicate
        more blurriness. Returns None if the image cannot be read.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return cv2.Laplacian(image, cv2.CV_64F).var()


def analyze_blurriness(directory):
    """
    Analyzes the blurriness of images in a directory, visualizes distribution,
    and saves severely blurry images.

    Args:
        directory (str): Path to the directory containing images.

    Returns:
        list: A list of tuples, each containing the image path and its
        blurriness score.
    """
    blurriness_scores = []

    for image_path in tqdm(Path(directory).rglob("*.jpeg"),
                           desc=f"Analyzing blurriness in {directory}"):
        score = compute_blurriness(image_path)
        if score is not None:
            blurriness_scores.append((str(image_path), score))

    scores = np.array([s[1] for s in blurriness_scores])

    plt.figure(figsize=(10, 5))
    plt.hist(scores, bins=50, color='blue', alpha=0.7)
    plt.axvline(np.percentile(scores, 25), color='red', linestyle='dashed',
                label="25th percentile (Most Blurry)")
    plt.axvline(np.percentile(scores, 50), color='green', linestyle='dashed',
                label="Median")
    plt.axvline(np.percentile(scores, 75), color='orange', linestyle='dashed',
                label="75th percentile (Least Blurry)")
    plt.xlabel("Laplacian Variance (Sharpness)")
    plt.ylabel("Number of Images")
    plt.title("Blurriness Distribution in Dataset")
    plt.legend()
    plt.show()

    blurry_threshold = np.percentile(scores, 25)
    blurry_images = [img for img,
                     score in blurriness_scores if score < blurry_threshold]

    with open("severely_blurry_images.txt", "w") as f:
        f.writelines("\n".join(blurry_images))

    print(f"Analysis complete! Saved {len(blurry_images)} images for review.")

    return blurriness_scores


def plot_blurriness_histogram(blurriness_scores):
    """
    Plots a histogram of blurriness scores for the images.

    Args:
        blurriness_scores (list): List of tuples containing image paths and
        their respective blurriness scores.

    Returns:
        None
    """
    scores = np.array([s[1] for s in blurriness_scores])

    max_sharpness = min(1000, np.max(scores))
    plt.figure(figsize=(10, 5))
    plt.hist(scores, bins=50, color='blue', alpha=0.7, range=(0,
                                                              max_sharpness))
    plt.axvline(np.percentile(scores, 25), color='red', linestyle='dashed',
                label="25th percentile (Most Blurry)")
    plt.axvline(np.percentile(scores, 50), color='green', linestyle='dashed',
                label="Median")
    plt.axvline(np.percentile(scores, 75), color='orange', linestyle='dashed',
                label="75th percentile (Least Blurry)")
    plt.xlabel("Laplacian Variance (Sharpness)")
    plt.ylabel("Number of Images")
    plt.title("Zoomed-In Blurriness Distribution")
    plt.legend()
    plt.show()


def get_image_hash(image_path):
    """
    Computes the MD5 hash of an image file.

    Args:
        image_path (str or Path): Path to the image file.

    Returns:
        str: The computed MD5 hash of the image.
    """
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def find_and_visualize_duplicates(directory):
    """
    Identifies duplicate images based on MD5 hashes, visualizes some of them,
    and deletes extra copies.

    Args:
        directory (str): Path to the directory containing images.

    Returns:
        None
    """
    hashes = defaultdict(list)

    for image_path in Path(directory).rglob("*.jpeg"):
        try:
            hash_value = get_image_hash(image_path)
            hashes[hash_value].append(image_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    duplicates = {h: p for h, p in hashes.items() if len(p) > 1}
    print(f"Found {len(duplicates)} sets of duplicate images.")

    for hash_value, image_paths in list(duplicates.items())[:5]:
        fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))
        fig.suptitle(f"Duplicate Set: {hash_value[:10]}...", fontsize=12)

        for ax, img_path in zip(axes, image_paths):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(str(img_path).split("/")[-1])

        plt.show()

    deleted_files = []
    for hash_value, image_paths in duplicates.items():
        for duplicate_img in image_paths[1:]:
            try:
                os.remove(duplicate_img)
                deleted_files.append(str(duplicate_img))
            except Exception as e:
                print(f"Error deleting {duplicate_img}: {e}")

    with open("deleted_duplicates.txt", "w") as f:
        f.writelines("\n".join(deleted_files))

    print(f"Deleted {len(deleted_files)} duplicate images.")
