# MedNIST Workflow

<!-- photo -->
![MedNIST Workflow](./image/MedNIST_dataset.png)

## Introduction

The MedNIST dataset is a collection of medical images from various modalities, including CT scans, X-rays, and MRI. It was created to facilitate research and educational purposes, providing a benchmark for image classification tasks in medical imaging. The dataset includes images categorized into six distinct classes:

- **AbdomenCT**
- **BreastMRI**
- **CXR** (Chest X-rays)
- **ChestCT**
- **Hand**
- **HeadCT**

I chose this project because medical imaging plays a crucial role in diagnosing and treating diseases. Leveraging machine learning to analyze medical images can significantly improve diagnostic accuracy and efficiency. The MedNIST dataset is an excellent starting point for exploring deep learning applications in healthcare, as it is well-organized and provides a diverse range of medical images.

This project implements two distinct workflows:

1. **Training Workflow**: The process of training a DenseNet121 model on the MedNIST dataset.
2. **Inference Workflow**: The process of using the trained model to classify unseen images.


## Steps to Reproduce

### 1. Prerequisites

1. **Install Nextflow**:

- Ensure you have Nextflow installed. If not, you can install it using the following command:
  ```bash
  curl -s https://get.nextflow.io | bash
  chmod +x nextflow
  sudo mv nextflow /usr/local/bin
  ```
- Verify Installation:

  ```bash
  nextflow -v
  ```

2. **Verify Docker Installation**:

- To ensure Docker is installed correctly, run:
  ```bash
  docker --version
  ```

- Ensure you have Nextflow installed. If not, you can install it using the following command:
  ```bash
  curl -s https://get.nextflow.io | bash
  ```

2. **Install Docker**:

   - Docker is required for this project. If you don’t have Docker installed, follow the instructions [here](https://docs.docker.com/get-docker/).

3. **Dependencies**:

   - Each workflow includes a `requirements.txt` file and a `Dockerfile` that lists all dependencies. However, the main ones are:
     - **Python 3.8.6**
     - **monai==1.3.2**
     - **scikit-learn==1.2.2**
     - **numpy==1.23.1**
     - **torch==1.13.1**
     - **scipy==1.9.0**
     - **Pillow==9.2.0**
     - **pandas==1.5.0**
     - **einops==0.8.0**
     - **transformers==4.46.3**
     - **matplotlib==3.1.3**

