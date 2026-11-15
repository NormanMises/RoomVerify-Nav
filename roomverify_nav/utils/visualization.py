# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import List

import cv2
import numpy as np


def add_text_to_image(image: np.ndarray, text: str, top: bool = False) -> np.ndarray:
    """
    Adds text to the given image.

    Args:
        image (np.ndarray): Input image.
        text (str): Text to be added.
        top (bool, optional): Whether to add the text to the top or bottom of the image.

    Returns:
        np.ndarray: Image with text added.
    """
    width = image.shape[1]
    text_image = generate_text_image(width, text)
    if top:
        combined_image = np.vstack([text_image, image])
    else:
        combined_image = np.vstack([image, text_image])

    return combined_image


def generate_text_image(width: int, text: str) -> np.ndarray:
    """
    Generates an image of the given text with line breaks, honoring given width.

    Args:
        width (int): Width of the image.
        text (str): Text to be drawn.

    Returns:
        np.ndarray: Text drawn on white image with the given width.
    """
    # Define the parameters for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    line_spacing = 10  # Spacing between lines in pixels

    # Calculate the maximum width and height of the text
    text_size, _ = cv2.getTextSize("Ag", font, font_scale, font_thickness)
    max_width = width - 20  # Allow some padding
    max_height = text_size[1] + line_spacing

    paragraphs = str(text).splitlines() or [str(text)]
    if not paragraphs:
        paragraphs = [""]

    x = 10
    y = text_size[1]
    to_draw = []
    num_rows = 0

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            num_rows += 1
            to_draw.append(("", x, y))
            y += max_height
            continue

        current_x = x
        current_y = y
        num_rows += 1
        for word in words:
            word_size, _ = cv2.getTextSize(word, font, font_scale, font_thickness)
            if current_x + word_size[0] > max_width and current_x > x:
                current_y += max_height
                current_x = x
                num_rows += 1
            to_draw.append((word, current_x, current_y))
            current_x += word_size[0] + 5
        y = current_y + max_height

    # Create a blank white image with the calculated dimensions
    image = 255 * np.ones((max_height * max(1, num_rows), width, 3), dtype=np.uint8)
    for word, x, y in to_draw:
        if not word:
            continue
        cv2.putText(
            image,
            word,
            (x, y),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )

    return image


def pad_images(images: List[np.ndarray], pad_from_top: bool = False) -> List[np.ndarray]:
    """
    Pads a list of images with white pixels to make them have the same dimensions.

    Args:
        images (List[np.ndarray]): List of NumPy images.
        pad_from_top (bool): If True, pad the images from the top; if False (default),
            pad from the bottom.

    Returns:
        List[np.ndarray]: List of padded images.

    """
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)

    padded_images = []
    for img in images:
        height_diff = max_height - img.shape[0]
        width_diff = max_width - img.shape[1]

        if pad_from_top:
            pad_top = height_diff
            pad_bottom = 0
        else:
            pad_top = 0
            pad_bottom = height_diff

        padded_img = np.pad(
            img,
            ((pad_top, pad_bottom), (0, width_diff), (0, 0)),
            mode="constant",
            constant_values=255,
        )
        padded_images.append(padded_img)

    return padded_images


if __name__ == "__main__":
    width = 400
    text = (
        "This is a long text that needs to be drawn on an image with a specified "
        "width. The text should wrap around if it exceeds the given width."
    )

    result_image = generate_text_image(width, text)

    cv2.imshow("Text Image", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
