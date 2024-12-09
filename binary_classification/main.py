# libraries
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler

# manual seed
torch.manual_seed(0)

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"We're using => {device}")

data_dir = "car_bike_dataset"
# print(f"data_dir: {data_dir}")

# define the transfroms
image_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
}

# load train data
train_dataset = datasets.ImageFolder(root=data_dir+"/train",
                                     transform=image_transforms["train"])

# load test data
test_dataset = datasets.ImageFolder(root=data_dir+"/test",
                                    transform=image_transforms["test"])


# get the size of the train_dataset and indices
train_dataset_size = len(train_dataset)
train_dataset_indices = list(range(train_dataset_size))

# shuffle the indices
np.random.shuffle(train_dataset_indices)

# get the splitter to split the train data to [TRAIN, VAL]
val_split_index = int(np.int32(0.2 * train_dataset_size))
train_idx, val_idx = train_dataset_indices[val_split_index:], train_dataset_indices[:val_split_index]

# use SubsetRandomSampler
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

# convert to DataLoader
train_dataloader = DataLoader(dataset=train_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=False, sampler=train_sampler)

val_dataloader = DataLoader(dataset=train_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=False, sampler=val_sampler)

test_dataloader = DataLoader(dataset=test_dataset, 
                             shuffle=False, batch_size=BATCH_SIZE)


# Build the Netword
class BinaryClassifierCNN(nn.Module):
    def __init__(self):
        super(BinaryClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=32 * 56 * 56, out_features=128)  
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

# define train function to to train and validate
def train(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)  # Ensure labels have the correct shape (N, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Get predictions
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.float().unsqueeze(1)  # Ensure labels have the correct shape (N, 1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Get predictions
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        # Calculate accuracy
        train_accuracy = correct_train / total_train
        val_accuracy = correct_val / total_val

        # Print statistics
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy*100:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy*100:.2f}%")


# define test function
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = (torch.sigmoid(outputs) > 0.5).int()
            correct += (predictions.squeeze() == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {correct / total * 100:.2f}%")

if __name__ == "__main__":
    model = BinaryClassifierCNN()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Use sigmoid with BCE
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train(model, train_dataloader, val_dataloader, criterion, optimizer, EPOCHS)
    test(model, test_dataloader)