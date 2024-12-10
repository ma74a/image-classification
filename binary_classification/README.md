
# Binary Classification with CNN

## Overview
This project implements a Convolutional Neural Network (CNN) in PyTorch for binary classification. The model classifies images into two categories (e.g., cars and bikes). The pipeline includes data preprocessing, model training, validation, and testing.

## Features
- **Data Handling**: 
  - Images are loaded using the `torchvision.datasets.ImageFolder` utility.
  - Data augmentation and transformation are applied using `torchvision.transforms`.
  - Train/validation splitting with `SubsetRandomSampler`.

- **Model Architecture**:
  - A CNN with two convolutional layers followed by max-pooling.
  - Fully connected layers for classification.
  - Activation functions: ReLU and Sigmoid.

- **Training and Evaluation**:
  - Binary Cross-Entropy with Logits Loss is used as the loss function.
  - Model performance is evaluated on both validation and test sets.
  - Accuracy metrics are reported for training, validation, and testing.

## Requirements
- Python 3.8+
- Libraries:
  - `torch`
  - `torchvision`
  - `numpy`
  - `matplotlib`
  - `tqdm`

Install dependencies with:
```bash
pip install torch torchvision numpy matplotlib tqdm
```

## Usage
### Dataset Structure
The dataset should be organized in the following format:
```
car_bike_dataset/
    train/
        car/
        bike/
    test/
        car/
        bike/
```

### Running the Project
1. Set the `data_dir` variable in the script to point to the dataset directory.
2. Run the script:
   ```bash
   python main.py
   ```
3. The script trains the model for 5 epochs (default) and evaluates it on the test dataset.

### Output
- **Training Logs**: Reports loss and accuracy for each epoch.
- **Test Accuracy**: Displays the final classification accuracy on the test set.

## Customization
- **Hyperparameters**: Modify `BATCH_SIZE`, `LEARNING_RATE`, and `EPOCHS` to tune the model.
- **Data Augmentation**: Add or modify transforms in `image_transforms`.
- **Model**: Extend the `BinaryClassifierCNN` class for more complex architectures.

## Example Output
```
Epoch 1/5, Train Loss: 0.4567, Train Accuracy: 85.67%, Val Loss: 0.3201, Val Accuracy: 89.34%
...
Test Accuracy: 91.23%
```
