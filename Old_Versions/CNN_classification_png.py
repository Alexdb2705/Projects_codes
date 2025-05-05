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

image_type = "Optic"     # Set which data type is to be used, between ISAR and Optic

if image_type == "ISAR" or image_type == "Optic":
    pass
else:
    raise ValueError("Data type, set at the beginning, must be established between ISAR and Optic.")

print(f"Using {device} device") 
print(f"Data type selected is {image_type}")

class CustomDataset(Dataset):
    def __init__(self, img1_dir, img2_dir, img3_dir, labels_file, transform=None):
        # Directorios de imágenes y archivo CSV de coordenadas
        self.img1_dir = img1_dir
        self.img2_dir = img2_dir
        self.img3_dir = img3_dir
        self.labels_vector = np.load(labels_file)

        self.transform = transform

    def __len__(self):
        return len(self.labels_vector)

    def __getitem__(self, idx):

        # Cargar las tres imágenes
        if image_type == "Optic":
            img1_path = os.path.join(self.img1_dir, f"Img_axis_x/sample_{idx+1}_x.png") #isar_{idx+1}_1.png
            img2_path = os.path.join(self.img2_dir, f"Img_axis_y/sample_{idx+1}_y.png")
            img3_path = os.path.join(self.img3_dir, f"Img_axis_z/sample_{idx+1}_z.png")
        elif image_type == "ISAR":
            img1_path = os.path.join(self.img1_dir, f"ISAR_x/isar_{idx+1}_1.png") #sample_{idx+1}_x.png
            img2_path = os.path.join(self.img2_dir, f"ISAR_y/isar_{idx+1}_2.png")
            img3_path = os.path.join(self.img3_dir, f"ISAR_z/isar_{idx+1}_3.png")
        img1 = Image.open(img1_path).convert("L")
        img2 = Image.open(img2_path).convert("L")
        img3 = Image.open(img3_path).convert("L")

        # Aplicar transformaciones si las hay
        if self.transform:
            img1 = self.transform(img1).to(device)
            img2 = self.transform(img2).to(device)
            img3 = self.transform(img3).to(device)

        # Obtener el vector de coordenadas de salida
        label = torch.tensor(self.labels_vector[idx], dtype=torch.float32, device=device)

        return img1, img2, img3, label

class CNNModule(nn.Module):
    def __init__(self):
        super(CNNModule, self).__init__()
        # Capas convolucionales
        self.conv1 = nn.LazyConv2d(96, kernel_size=22, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.LazyConv2d(256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.LazyConv2d(384, kernel_size=3, padding=1)
        self.conv4 = nn.LazyConv2d(384, kernel_size=3, padding=1)
        self.conv5 = nn.LazyConv2d(256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
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
    def __init__(self, num_labels):
        super(Geometry_Predictor, self).__init__()
        # CNN compartida para las tres imágenes
        self.cnn = CNNModule()

        # Fully connected layers
        self.fc1 = nn.LazyLinear(4096)  # Ajustar el tamaño si las imágenes son diferentes
        self.fc2 = nn.LazyLinear(4096)
        #self.Dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.LazyLinear(num_labels)  # tantas salidas como stl de los que existen fotos en el dataset

    def forward(self, image1, image2, image3):
        # Procesar cada imagen a través de la CNN
        image1_features = self.cnn(image1)
        image2_features = self.cnn(image2)
        image3_features = self.cnn(image3)

        # Concatenar las características de las tres imágenes
        concat_images = torch.cat((image1_features, image2_features, image3_features), dim=1)

        # Pasar por las capas densas
        x = F.relu(self.fc1(concat_images))
        #x = self.Dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.Dropout(x)
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

# Definir las transformaciones (por ejemplo, redimensionar las imágenes y normalizarlas)
transform = transforms.Compose([
    transforms.Resize((128, 128)),   # Cambia esto según el tamaño de tus imágenes
    transforms.ToTensor(),   # Convertir las imágenes a tensores
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalización
])

# Crear el dataset personalizado
dataset = CustomDataset(
    img1_dir=f"./Img/{image_type}",
    img2_dir=f"./Img/{image_type}",
    img3_dir=f"./Img/{image_type}",
    coords_file=f"./Img/coords.npy",
    transform=transform
)

# Dividir el dataset en entrenamiento y validación
train_size = int(0.7 * len(dataset))  # 70% para entrenamiento
val_size = int(0.15 * len(dataset))  # 15% para validación
test_size = len(dataset) - train_size - val_size  # 15% para test

generator = torch.Generator().manual_seed(101)

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Crear DataLoaders para iterar sobre los datasets
train_batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# Inicializar modelo
model = Geometry_Predictor(len(np.unique(dataset.labels_vector))).to(device)

# Early Stopping Instance
early_stopping = EarlyStopping(patience=10)

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
    for img1, img2, img3, target_label in train_loader:
        optimizer.zero_grad()

        # Forward pass
        output_logits = model(img1, img2, img3) # This output gives the logits, not the probabilities
        pred_probs= F.softmax(output_logits, dim=1) # This converts the logits to probabilities

        loss = criterion_CEL(output_logits, target_label.long())
        running_train_loss += loss.item()

        # Backward pass y optimización
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        for img1, img2, img3, target_label in val_loader:
            
            output_logits = model(img1, img2, img3)
            pred_probs = F.softmax(output_logits, dim=1)

            loss = criterion_CEL(output_logits, target_label.long())
            avg_val_loss[epoch] = loss.item()
        
    # Check Early Stopping
    if early_stopping(avg_val_loss[epoch]):
        break  # Stop training

    avg_train_loss[epoch] = running_train_loss / len(train_loader)
    end_time = time.time()
    epoch_duration = end_time - start_time
    
    print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_duration:.2f} seconds with a training error of {avg_train_loss[epoch]:.4f} and a validation loss of {avg_val_loss[epoch]:.4f}")

print("Entrenamiento completado")

# Guardar el modelo

torch.save(model.state_dict(), f"classifier_1500samples_100epochs_32bs.pth")

# Evaluación en el conjunto de test
test_loss = 0.0
model.eval()
with torch.no_grad():
    for img1, img2, img3, target_label in test_loader:
        
        output_logits = model(img1, img2, img3)
        pred_probs = F.softmax(output_logits, dim=1)
        loss = criterion_CEL(output_logits, target_label.long())
        
        test_loss = loss.item()

print("Error medio de test: ", test_loss)

