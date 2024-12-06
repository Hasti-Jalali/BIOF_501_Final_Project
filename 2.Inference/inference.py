import os
import torch
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import DenseNet121

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class InferenceDataset(Dataset):
    def __init__(self, image_dict, transforms=None):
        """
        Dataset for inference.

        Args:
            image_dict (dict): Dictionary of image names and tensors.
            transforms (callable, optional): Optional transforms to apply on the images.
        """
        self.image_names = list(image_dict.keys())
        self.image_tensors = list(image_dict.values())
        self.transforms = transforms

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = self.image_tensors[index]
        if self.transforms:
            image = self.transforms(image)
        return self.image_names[index], image


def run_test(preprocessed_file, model_path, temp_output_file, label_names):
    """
    Runs inference on preprocessed images and saves results to a temporary file.

    Args:
        preprocessed_file (str): Path to preprocessed images file (.pt).
        model_path (str): Path to the trained model file (.pth).
        temp_output_file (str): Path to save intermediate results.
        label_names (list): List of class label names.
    """
    # Load preprocessed images
    image_dict = torch.load(preprocessed_file)

    # Create dataset and dataloader
    dataset = InferenceDataset(image_dict)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=2)

    # Load model
    num_classes = len(label_names)
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Run inference
    results = []
    with torch.no_grad():
        for batch in dataloader:
            image_names, images = batch
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            for img_name, pred in zip(image_names, predictions):
                results.append([img_name, label_names[pred.item()]])

    # Save intermediate results
    torch.save(results, temp_output_file)
    print(f"Inference results saved to temporary file '{temp_output_file}'.")


if __name__ == "__main__":
    # Define paths and labels
    preprocessed_file = "preprocessed_test_images.pt"
    model_path = "best_metric_model.pth"
    temp_output_file = "results.pt"  # Temporary file
    label_names = ["AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT"]

    run_test(preprocessed_file, model_path, temp_output_file, label_names)
