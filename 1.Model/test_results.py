import os
import torch
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Activations, AsDiscrete
from monai.networks.nets import DenseNet121

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class MedNISTDataset(Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]

def test_model(data_dir, model_path, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load test split and transforms
    test_split = torch.load(os.path.join(data_dir, "test_split.pt"))
    val_test_transforms = torch.load(os.path.join(data_dir, "val_test_transforms.pt"))

    # Create dataset and loader
    test_ds = MedNISTDataset(test_split["images"], test_split["labels"], val_test_transforms)
    test_loader = DataLoader(test_ds, batch_size=300, num_workers=2)

    # Load model
    num_class = len(set(test_split["labels"]))
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_class).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluate
    y_true = []
    y_pred = []
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())

    # Classification report
    class_names = ["AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT"]  # Update if necessary
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

if __name__ == "__main__":
    test_model(
        data_dir="data/preprocessed",
        model_path="models/best_metric_model.pth",
        output_dir="results/test"
    )
