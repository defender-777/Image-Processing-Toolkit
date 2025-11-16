"""
augmentation.py
---------------
Performs various image augmentation techniques to expand datasets.
Includes rotation, flipping, scaling, brightness changes, and translations.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def rotate_image(img, angle):
    """Rotate image by a given angle."""
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h))
    return rotated

def flip_image(img, mode=1):
    """Flip image horizontally or vertically."""
    return cv2.flip(img, mode)

def adjust_brightness(img, value=30):
    """Adjust brightness by adding value to all pixels."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v + value, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    bright_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return bright_img

def translate_image(img, x_shift, y_shift):
    """Translate image by x and y pixels."""
    matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    shifted = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
    return shifted

if __name__ == "__main__":
    img = cv2.imread("lenna.png")
    if img is None:
        print("❌ Image not found. Place 'lenna.png' in this folder.")
        exit()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    augmentations = {
        "Rotated": rotate_image(img, random.choice([15, -15, 30])),
        "Flipped": flip_image(img, 1),
        "Brightened": adjust_brightness(img, 40),
        "Translated": translate_image(img, 40, 30)
    }

    os.makedirs("output", exist_ok=True)

    # Display & Save
    fig, ax = plt.subplots(1, len(augmentations) + 1, figsize=(18, 6))
    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[0].axis("off")

    for i, (name, aug_img) in enumerate(augmentations.items(), 1):
        ax[i].imshow(aug_img)
        ax[i].set_title(name)
        ax[i].axis("off")
        cv2.imwrite(f"output/{name.lower()}.png", cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

    plt.tight_layout()
    plt.show()
