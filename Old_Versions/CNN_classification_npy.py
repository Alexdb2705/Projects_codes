import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
import  matplotlib.pyplot   as plt
import math
# from torchsummary import summary
import csv
import time

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
         )

torch.manual_seed(101)
torch.cuda.manual_seed(101)

print(f"\nUsing {device} device\n") 

class CustomDataset(Dataset):
    """Custom dataset for loading .npy files and their corresponding labels.

    Initializes the CustomDataset instance.

    Args:
        npy_dir (str): Directory containing the .npy files.
        labels_file (str): File path to the .npy file containing labels.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, npy_dir, labels_file, transform=None):
       
        self.npy_dir = npy_dir
        
        # List all .npy files in the directory.
        self.file_names = [f for f in os.listdir(npy_dir) if f.endswith('.npy') and f.startswith('sample_')]
        
        # Sort files based on the sample id which is the number right after "sample_"
        # For example, "sample_5_322.806383_T.npy" will be split into
        # ['sample', '5', '322.806383', 'T.npy'].
        self.file_names.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        # Load labels vector from the labels file.
        self.labels_vector = np.load(labels_file)
        
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.file_names)

    def __getitem__(self, idx):
        """Retrieves a sample and its corresponding label by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple (data, label) where:
                - data (np.ndarray): Loaded numpy array from the .npy file (dimensions n x m x 2).
                - label (torch.Tensor): Corresponding label as a float32 tensor.
        """
        # Construct the full file path for the sample.
        file_path = os.path.join(self.npy_dir, self.file_names[idx])
        
        # Load the numpy array with shape (n x m x 2).
        data = np.load(file_path)
        
        # Convert the numpy array to a PyTorch tensor.
        data = torch.tensor(data, dtype=torch.float32).to(device)
        
        # Convert the corresponding label to a torch tensor with dtype float32.
        label = torch.tensor(self.labels_vector[idx], dtype=torch.float32).to(device)
        
        return data, label
    
class CNNModule(nn.Module):
    """Convolutional module to extract features from a 2-channel input.

    This network processes an input tensor with shape (batch_size, 2, height, width)
    and returns a flattened feature vector.
    """

# La red LSTM podria usarse para relacionar columnas o filas del npy entre sí

    def __init__(self):
        super(CNNModule, self).__init__()
        # Capas convolucionales
        self.conv1 = nn.LazyConv2d(96, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.LazyConv2d(256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.conv3 = nn.LazyConv2d(384, kernel_size=3, padding=1)
        self.conv4 = nn.LazyConv2d(384, kernel_size=3, padding=1)
        self.conv5 = nn.LazyConv2d(256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.ReLU = nn.ReLU()
        self.Flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.ReLU(x)
        x = self.conv4(x)
        x = self.ReLU(x)
        x = self.conv5(x)
        x = self.ReLU(x)
        x = self.pool3(x)
        x = self.Flatten(x)
        return x
    
class Geometry_Predictor(nn.Module):
    """Geometry predictor model that processes a single 2-channel input.

    The model extracts features using a CNN and then passes them through fully
    connected layers to produce predictions.
    """

    def __init__(self, num_labels):
        """Initializes the Geometry_Predictor.

        Args:
            num_labels (int): Number of output labels.
        """
        super(Geometry_Predictor, self).__init__()
        # CNN compartida para las tres imágenes
        self.cnn = CNNModule()

        # Fully connected layers
        self.fc1 = nn.LazyLinear(4096)  # Adjust size if needed.
        self.fc2 = nn.LazyLinear(4096)
        #self.Dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.LazyLinear(num_labels)  # Number of outputs equal to num_labels.

    def forward(self, x):
        """Forward pass for the geometry predictor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2, height, width).

        Returns:
            torch.Tensor: Model output.
        """
        # Process the input through the CNN to extract features.
        features = self.cnn(x)
        # Pass the features through fully connected layers.
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        output_prob = self.fc3(x)
        return output_prob

class EarlyStopping:

    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait before stopping if no improvement.
            min_delta (float): Minimum change in loss to be considered as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.patience} epochs.")
                return True
        return False

# Crear el dataset personalizado
dataset = CustomDataset(
    npy_dir=f"/home/newfasant/N101-IA/Datasets/Reorganized/Classification_500_0_64_f_64_d",
    labels_file=f"/home/newfasant/N101-IA/Datasets/Reorganized/Classification_500_0_64_f_64_d/labels_vector_500_0_64_f_64_d.npy"
)

# Dividir el dataset en entrenamiento y validación
train_size = int(0.7 * len(dataset))  # 70% para entrenamiento
val_size = int(0.15 * len(dataset))  # 15% para validación
test_size = len(dataset) - train_size - val_size  # 15% para test

generator = torch.Generator().manual_seed(101)

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Crear DataLoaders para iterar sobre los datasets
train_batch_size = 2
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# Inicializar modelo
model = Geometry_Predictor(len(np.unique(dataset.labels_vector))).to(device)

# Early Stopping Instance
early_stopping = EarlyStopping(patience=100)

# Definir el optimizador y la función de pérdida
optimizer = optim.SGD(params=model.parameters(), lr=0.01)
criterion_CEL = nn.CrossEntropyLoss()

# Entrenamiento
num_epochs = 100
avg_val_loss = np.empty(num_epochs, dtype=np.float32)
avg_train_loss = np.empty(num_epochs, dtype=np.float32)
avg_val_loss = np.empty(num_epochs, dtype=np.float32)

for epoch in range(num_epochs):
    
    start_time = time.time()
    running_train_loss = 0.0
    
    model.train()
    for npy, target_label in train_loader:
        optimizer.zero_grad()

        # Forward pass
        output_logits = model(npy) # This output gives the logits, not the probabilities
        pred_probs= F.softmax(output_logits, dim=1) # This converts the logits to probabilities

        loss = criterion_CEL(output_logits, target_label.long())
        running_train_loss += loss.item()
        
        # print(npy)
        # print(npy.shape)
        # print(target_label)
        # print(output_logits)
        # print(pred_probs)
        print(loss.item())

        # Backward pass y optimización
        loss.backward()
        optimizer.step()
    
    print(running_train_loss)
    exit()

    model.eval()
    with torch.no_grad():
        for npy, target_label in val_loader:
            
            output_logits = model(npy)
            pred_probs = F.softmax(output_logits, dim=1)

            loss = criterion_CEL(output_logits, target_label.long())
            avg_val_loss[epoch] = loss.item()
    
    avg_train_loss[epoch] = running_train_loss / len(train_loader)

    # Check Early Stopping
    if early_stopping(avg_val_loss[epoch]):
        break  # Stop training

    end_time = time.time()
    epoch_duration = end_time - start_time
    
    print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_duration:.2f} seconds with a training error of {avg_train_loss[epoch]:.4f} a validation loss of {avg_val_loss[epoch]:.4f}")

print("Entrenamiento completado")

# Guardar el modelo

torch.save(model.state_dict(), f"/home/newfasant/N101-IA/CNN/Models")

# Evaluación en el conjunto de test
test_loss = 0.0
model.eval()
with torch.no_grad():
    for npy, target_label in test_loader:
        
        output_logits = model(npy)
        pred_probs = F.softmax(output_logits, dim=1)
        loss = criterion_CEL(output_logits, target_label.long())
        
        test_loss = loss.item()

print("Error medio de test: ", test_loss)

