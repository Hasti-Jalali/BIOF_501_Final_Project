import os
import torch
from torch.utils.data import DataLoader, Dataset
from monai.transforms import Activations, AsDiscrete
from monai.networks.nets import DenseNet121
from monai.metrics import ROCAUCMetric
from torch.optim import Adam
import matplotlib.pyplot as plt

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

def train_model(data_dir, output_dir, model_path):
    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Load splits and transforms
    train_split = torch.load(os.path.join(data_dir, "train_split.pt"))
    val_split = torch.load(os.path.join(data_dir, "val_split.pt"))
    train_transforms = torch.load(os.path.join(data_dir, "train_transforms.pt"))
    val_test_transforms = torch.load(os.path.join(data_dir, "val_test_transforms.pt"))

    # Create datasets and loaders
    train_ds = MedNISTDataset(train_split["images"], train_split["labels"], train_transforms)
    val_ds = MedNISTDataset(val_split["images"], val_split["labels"], val_test_transforms)
    train_loader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=300, num_workers=2)

    # Model, loss, optimizer, and metrics
    num_class = len(set(train_split["labels"]))
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_class).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-5)
    act = Activations(softmax=True)
    to_onehot = AsDiscrete(to_onehot=num_class)
    auc_metric = ROCAUCMetric()
    best_metric = -1
    best_metric_epoch = -1

    # Training settings
    epoch_num = 4
    val_interval = 1
    epoch_loss_values = []
    metric_values = []

    # Training loop
    for epoch in range(epoch_num):
        print('-' * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                y_onehot = [to_onehot(i) for i in y]
                y_pred_act = [act(i) for i in y_pred]
                auc_metric(y_pred_act, y_onehot)
                auc_result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot
                metric_values.append(auc_result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                if acc_metric > best_metric:
                    best_metric = acc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), model_path)
                    print(f"Saved new best metric model at epoch {epoch + 1}")

                print(f"current epoch: {epoch + 1} current AUC: {auc_result:.4f}"
                      f" current accuracy: {acc_metric:.4f} best AUC: {best_metric:.4f}"
                      f" at epoch: {best_metric_epoch}")
    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    # Save loss and validation plots
    plt.figure('train', (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel('epoch')
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Validation: Area under the ROC curve")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel('epoch')
    plt.plot(x, y)
    plt.savefig(os.path.join(output_dir, "training_plots.png"))
    plt.show()
    print("Training metrics plot saved.")

if __name__ == "__main__":
    train_model(data_dir="/app/data/preprocessed", 
                output_dir="/app/results/train", 
                model_path="/app/models/best_metric_model.pth")
