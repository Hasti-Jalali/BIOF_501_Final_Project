import os
import torch
import numpy as np
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ToTensor,
)
from PIL import Image

def preprocess_data(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Define transforms
    train_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        RandRotate(range_x=15, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        ToTensor()
    ])
    val_test_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        ToTensor()
    ])

    # Save transforms
    torch.save(train_transforms, os.path.join(output_dir, "train_transforms.pt"))
    torch.save(val_test_transforms, os.path.join(output_dir, "val_test_transforms.pt"))



    class_names = sorted([x for x in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, x))])
    num_class = len(class_names)
    image_files = [[os.path.join(input_dir, class_name, x)
                    for x in os.listdir(os.path.join(input_dir, class_name))]
                for class_name in class_names]
    image_file_list = []
    image_label_list = []
    for i, class_name in enumerate(class_names):
        image_file_list.extend(image_files[i])
        image_label_list.extend([i] * len(image_files[i]))
    num_total = len(image_label_list)
    image_width, image_height = Image.open(image_file_list[0]).size

    print('Total image count:', num_total)
    print("Image dimensions:", image_width, "x", image_height)
    print("Label names:", class_names)
    print("Label counts:", [len(image_files[i]) for i in range(num_class)])

    valid_frac, test_frac = 0.1, 0.1
    trainX, trainY = [], []
    valX, valY = [], []
    testX, testY = [], []

    for i in range(num_total):
        rann = np.random.random()
        if rann < valid_frac:
            valX.append(image_file_list[i])
            valY.append(image_label_list[i])
        elif rann < test_frac + valid_frac:
            testX.append(image_file_list[i])
            testY.append(image_label_list[i])
        else:
            trainX.append(image_file_list[i])
            trainY.append(image_label_list[i])

    print("Training count =",len(trainX),"Validation count =", len(valX), "Test count =",len(testX))

    # Save splits
    torch.save({"images": trainX, "labels": trainY}, os.path.join(output_dir, "train_split.pt"))
    torch.save({"images": valX, "labels": valY}, os.path.join(output_dir, "val_split.pt"))
    torch.save({"images": testX, "labels": testY}, os.path.join(output_dir, "test_split.pt"))

if __name__ == "__main__":
    preprocess_data(input_dir="/app/data/MedNIST", output_dir="/app/data/preprocessed")
