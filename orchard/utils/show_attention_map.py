# TODO
import cv2
import numpy as np


def display_attention_map(image_path, attention_map):
    # Load the original image
    original_image = cv2.imread(image_path)

    # Ensure the attention map is normalized between 0 and 1
    attention_map_normalized = cv2.normalize(
        attention_map,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )

    # Resize the attention map to match the dimensions of the original image
    attention_resized = cv2.resize(
        attention_map_normalized, (original_image.shape[1], original_image.shape[0])
    )

    # Convert the normalized attention map to uint8 (range 0-255) for visualization
    attention_uint8 = np.uint8(255 * attention_resized)

    # Apply a color map (e.g., JET) to the attention map for better visualization
    attention_colored = cv2.applyColorMap(attention_uint8, cv2.COLORMAP_JET)

    # Blend the original image with the attention map using alpha blending
    alpha = 0.6  # Transparency factor for the attention map
    blended_image = cv2.addWeighted(
        original_image, 1 - alpha, attention_colored, alpha, 0
    )

    # Display the blended image with the attention map overlay
    cv2.imshow("Attention Map", blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage:
# Replace 'image.jpg' with your actual image path and provide an attention map (numpy array)
# attention_map = np.random.rand(14, 14)  # Random example attention map
# display_attention_map('image.jpg', attention_map)
