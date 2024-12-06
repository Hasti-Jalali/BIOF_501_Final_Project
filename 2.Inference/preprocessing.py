import os
import torch
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensity,
    ToTensor,
    Compose,
)

def preprocess_test_images(input_dir, output_file):
    """
    Preprocess images in the test_mednist folder.

    Args:
        input_dir (str): Path to the folder containing test images.
        output_file (str): Path to save preprocessed images as a .pt file.

    Returns:
        None
    """
    # Define validation/test transforms
    val_test_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        ToTensor()
    ])

    # Process and save each image
    preprocessed_images = {}
    for image_name in sorted(os.listdir(input_dir)):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, image_name)
            transformed_image = val_test_transforms(image_path)  # Apply transforms
            preprocessed_images[image_name] = transformed_image

    # Save all preprocessed images as a PyTorch tensor dictionary
    torch.save(preprocessed_images, output_file)
    print(f"Preprocessed test images saved to {output_file}")


if __name__ == "__main__":
    preprocess_test_images(input_dir="test_mednist", output_file="preprocessed_test_images.pt")
