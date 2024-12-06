# MedNIST Workflow


## Background and Rationale

### What’s and Why’s

The MedNIST workflow leverages deep learning to address the growing need for efficient and accurate medical image classification. This project aligns with the pressing challenge of improving diagnostic workflows in healthcare. Automating image classification can enhance decision-making, reduce the burden on medical professionals, and improve patient outcomes.

1. **Importance in Healthcare**:
   - Medical imaging is essential for diagnosing a wide range of diseases, including cancer, cardiovascular conditions, and neurological disorders.
   - Automating classification tasks enables healthcare professionals to focus on critical cases, ensuring timely and accurate diagnoses, especially in resource-limited settings.
   - Automation also reduces diagnostic errors by providing consistent results unaffected by fatigue or subjective bias.

2. **Why the MedNIST Dataset?**:
   - The MedNIST dataset provides a reliable benchmark for exploring machine learning applications in medical imaging.
   - Its six classes (AbdomenCT, BreastMRI, CXR, ChestCT, Hand, HeadCT) represent a diverse array of imaging modalities, enabling the creation of generalized and robust models.

3. **Real-World Applications**:
   - Prioritizing high-risk cases for rapid human review.
   - Supporting medical professionals in interpreting complex or ambiguous images.
   - Extending diagnostic capabilities to remote or underserved regions with limited access to specialists.

<!-- photo -->
![MedNIST Workflow](./image/MedNIST_dataset.png)

---

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

---
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

2. **Install Docker**:

   - Docker is required for this project. If you don’t have Docker installed, follow the instructions [here](https://docs.docker.com/get-docker/).

- To ensure Docker is installed correctly, run:
  ```bash
  docker --version
  ```

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

---

## Running the Project

### Running the First Workflow: Training

1. **Navigate to the `1.Model` directory**:

     ```bash
     cd 1.Model
     ```

2. **Verify Nextflow Installation**:
   - Verify that Nextflow is installed by running:
     ```bash
     nextflow -v
     ```
   - If Nextflow is not installed, follow the steps in the prerequisites section to install it.

3. **Build the Docker Image**:

   - Create the Docker image for the training workflow using the following command:
     ```bash
     docker build -t mednist_training:latest .
     ```

4. **Run Nextflow**:

   - Execute the training workflow using Nextflow:
     ```bash
     nextflow run main.nf --with-docker
     ```

5. **Output of the Training Workflow**:

   - The output of this workflow is saved in the following directories:

     - `./results/train`
     - `./results/test`

   - **Training Metrics Plot**:

     - A file named `training_plots.png` is generated, containing two plots:
       1. **Epoch Average Loss**:
          - This plot shows the average loss per epoch, illustrating how the model’s training loss decreases over time, indicating improved learning.
       2. **Validation Area Under the ROC Curve**:
          - This plot shows the ROC AUC for the validation dataset per epoch, demonstrating how well the model distinguishes between classes during validation.

    ![Training plot](./image/training_plots.png)

   - **Test Metrics Plot**:

     The output of the testing workflow includes a file named `classification_report.txt`.

     **Classification Report**:

     The report contains precision, recall, F1-score, and support for each class. For example:
    
    ![Test plot](./image/test_report.png)

     **Explanation of Metrics**:

     - **Precision**: The ratio of true positives to the sum of true and false positives for each class.
     - **Recall**: The ratio of true positives to the sum of true positives and false negatives for each class.
     - **F1-Score**: The harmonic mean of precision and recall, providing a single metric for classification performance.
     - **Support**: The number of true instances for each class.

   **Explanation of Each Step**:

     #### Step 1: Download Data
     - **What Happens**:  
     The `data_gathering.sh` script downloads the MedNIST dataset from a predefined URL and extracts it into the `data`  directory.  
     - **Key Details**:  
     - The URL (`https://www.dropbox.com/s/5wwskxctvcxiuea/MedNIST.tar.gz`) points to a compressed archive containing the dataset.
     - The script uses `curl` to download the file and `tar` to extract it.
     - The extracted files are stored in `data/MedNIST`, and the archive is deleted to save space.
     - **Why This is Important**:  
     This step ensures all necessary data is readily available for preprocessing and training.

     #### Step 2: Preprocess Data
     - **What Happens**:  
     The `data_preprocessing.py` script processes images and prepares the dataset for model training.  
     - **Key Details**:  
     - **Augmentation**: Adds variability to the data with techniques like random flipping, rotation, and zooming (`RandFlip`, `RandRotate`, `RandZoom`).
     - **Normalization**: Scales pixel intensity values between 0 and 1 to standardize inputs.
     - **Splitting**: Divides the dataset into training, validation, and test subsets with proportions controlled by `valid_frac` and `test_frac`.
     - **Outputs**:  
         - Transformed data is saved as PyTorch tensors in `data/preprocessed`:
        - `train_split.pt`, `val_split.pt`, `test_split.pt` for splits.
        - `train_transforms.pt`, `val_test_transforms.pt` for reusable transformations.
     - **Why This is Important**:  
     Proper preprocessing ensures the model sees standardized and augmented data, improving learning and generalization.

     #### Step 3: Train the Model
     - **What Happens**:  
     The `train_validation.py` script trains a DenseNet121 model using the processed data.  
     - **Key Details**:  
     - **Training Loop**: 
         - Iteratively adjusts model weights using `Adam` optimizer and `CrossEntropyLoss`.
        - Tracks loss at each step and epoch.
     - **Validation**:  
         - Evaluates the model after every epoch, calculating metrics like accuracy and AUC.
     - **Outputs**:
         - The best-performing model is saved in `models/best_metric_model.pth`.
         - A plot (`training_plots.png`) visualizes:
         - Training loss over epochs.
         - Validation AUC over epochs.
     - **Why This is Important**:  
     This step builds the classification model and ensures it achieves optimal performance.

     #### Step 4: Evaluate the Model
     - **What Happens**:  
     The `test_results.py` script evaluates the trained model on the test dataset.  
     - **Key Details**:  
     - **Metrics**: Generates precision, recall, and F1-scores for each class.
     - **Outputs**: Saves the evaluation results in `results/test/classification_report.txt`.
     - **Why This is Important**:  
     Provides insights into how well the model performs on unseen data.


### Running the Second Workflow: Inference

1.  **Navigate to the `2.Inference` directory**:

     ```bash
     cd 2.Inference
     ```

2. **Verify Nextflow Installation**:

   - Verify that Nextflow is installed by running:
     ```bash
     nextflow -v
     ```
   - If Nextflow is not installed, follow the steps in the prerequisites section to install it.

3. **Build the Docker Image**:

   - Create the Docker image for the inference workflow using the following command:
     ```bash
     docker build -t mednist_inference:latest .
     ```

4. **Run Nextflow**:

   - Execute the inference workflow using Nextflow:
     ```bash
     nextflow run main.nf --with-docker
     ```


5. **Output of the Inference Workflow**:

   - The output of this workflow is a file named `test_predictions.csv`, which contains the classifications for the images in the `./test_mednist` folder.


   **Explanation of Each Step**:

    #### Step 1: Read and Preprocess Image Data
     - **What Happens**:
     Images from the `./test_mednist` directory are read and preprocessed using the `preprocessing.py` script. This ensures that the test images undergo the same transformations as the training data to maintain consistency.
     - **Details**:
     - Images are loaded and normalized to standardize pixel intensity values.
     - Transforms such as `ToTensor` convert the image data into PyTorch-compatible formats.

    #### Step 2: Classify Images
     - **What Happens**:
     A pretrained DenseNet121 model is used to predict the classes of the preprocessed images. This model can either be obtained from the training workflow or provided as a standalone file.
     - **Details**:
     - The model assigns probabilities to each class for every image.
     - The class with the highest probability is selected as the predicted label.

    #### Step 3: Generate CSV Output
     - **What Happens**:
     A CSV file named `test_predictions.csv` is created to store the results of the inference process.
     - **Details**:
     - The CSV file contains two columns:
         - **Image Name**: The name of each processed image file.
         - **Predicted Label**: The label assigned by the model.

     **Outputs**:
     - The results of the inference workflow are saved in the `test_predictions.csv` file, which facilitates easy validation and interpretation of the predictions. 

     **[The test_predictions.csv file, which serves as the output of this workflow, is saved in the current directory.]**


    **Explanation**:
    - The Docker image encapsulates all dependencies for inference.
    - Nextflow provides an easy-to-use interface to run the inference pipeline consistently across environments.
---

