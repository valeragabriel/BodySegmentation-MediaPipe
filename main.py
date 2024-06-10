import cv2
import mediapipe as mp
import numpy as np
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def resize_and_show(image):
    DESIRED_HEIGHT = 480
    DESIRED_WIDTH = 480
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_segmented_image(image_path):
    # Create the options that will be used for ImageSegmenter
    base_options = python.BaseOptions(model_asset_path='selfie_multiclass_256x256.tflite')
    options = vision.ImageSegmenterOptions(base_options=base_options,
                                           output_category_mask=True)

    # Create the image segmenter
    with vision.ImageSegmenter.create_from_options(options) as segmenter:

        # Load the image
        image = mp.Image.create_from_file(image_path)

        # Retrieve the masks for the segmented image
        segmentation_result = segmenter.segment(image)
        category_mask = segmentation_result.category_mask.numpy_view()

        # Load the original image
        original_image = cv2.imread(image_path)

        # Create a white image of the same size as the original image
        white_background = np.ones_like(original_image) * 255

        # Apply the mask to the original image
        segmented_image = np.where(np.expand_dims(category_mask, axis=-1) > 0.2, original_image, white_background)

        return segmented_image


if __name__ == "__main__":
    
    image_path = 'imagem_ref.png'

    # Process the segmented image
    segmented_image = process_segmented_image(image_path)

    # Save the segmented image
    output_path = 'segmented_image.png'
    cv2.imwrite(output_path, segmented_image)

    # Display the segmented image
    resize_and_show(segmented_image)
