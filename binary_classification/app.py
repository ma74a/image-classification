import streamlit as st
from PIL import Image
import torch.nn as nn
import torch
from torchvision import transforms

class BinaryClassifierCNN(nn.Module):
    def __init__(self):
        super(BinaryClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # Adjust based on input size
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def load_model():
    model = BinaryClassifierCNN()
    model.load_state_dict(torch.load('binary_cnn_3.pth'))
    model.eval()
    return model

def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    output = model(image)
    prediction = torch.sigmoid(output).item()
    return prediction > 0.5

# Streamlit app
st.title("Binary Classifier")
model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    result = predict(image, model)
    st.write(f"Prediction: {'Car' if result else 'Bike'}")
