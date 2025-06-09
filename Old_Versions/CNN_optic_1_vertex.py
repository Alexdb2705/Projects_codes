import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
import  matplotlib.pyplot   as plt
import math
from torchsummary import summary

device='cuda' if torch.cuda.is_available() else 'cpu'

class CustomDataset(Dataset):
    def __init__(self, img1_dir, img2_dir, img3_dir, coords_file, transform=None):
        # Directorios de imágenes y archivo CSV de coordenadas
        self.img1_dir = img1_dir
        self.img2_dir = img2_dir
        self.img3_dir = img3_dir
        self.coords = pd.read_csv(coords_file, sep=';', dtype={'x': np.float32, 'y': np.float32, 'z':np.float32})
        print(self.coords)
        self.transform = transform

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        # Obtener el nombre de la muestra
        sample_name = self.coords.iloc[idx, 0]

        # Cargar las tres imágenes
        img1_path = os.path.join(self.img1_dir, f"{sample_name}_x.png")
        img2_path = os.path.join(self.img2_dir, f"{sample_name}_y.png")
        img3_path = os.path.join(self.img3_dir, f"{sample_name}_z.png")
        img1 = Image.open(img1_path).convert("L")
        img2 = Image.open(img2_path).convert("L")
        img3 = Image.open(img3_path).convert("L")

        # Aplicar transformaciones si las hay
        if self.transform:
            img1 = self.transform(img1).to(device)
            img2 = self.transform(img2).to(device)
            img3 = self.transform(img3).to(device)

        # Obtener el vector de coordenadas de salida
        coords = torch.tensor(np.array(self.coords.iloc[idx, 1:].values, dtype=np.float32), dtype=torch.float32)

        return img1, img2, img3, coords

class CNNModule(nn.Module):
    def __init__(self):
        super(CNNModule, self).__init__()
        # Capas convolucionales
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)
        return x

class CoordinatePredictor(nn.Module):
    def __init__(self):
        super(CoordinatePredictor, self).__init__()
        # CNN compartida para las tres imágenes
        self.cnn = CNNModule()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128 * 8 * 8 * 3, out_features=128)  # Ajustar el tamaño si las imágenes son diferentes
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Tres salidas, una por coordenada (x, y, z)

    def forward(self, image1, image2, image3):
        # Procesar cada imagen a través de la CNN
        image1_features = self.cnn(image1)
        image2_features = self.cnn(image2)
        image3_features = self.cnn(image3)

        # Concatenar las características de las tres imágenes
        concat_images = torch.cat((image1_features, image2_features, image3_features), dim=1)

        # Pasar por las capas densas
        x = F.relu(self.fc1(concat_images))
        x = F.relu(self.fc2(x))
        output_coords = self.fc3(x)

        return output_coords

# Definir las transformaciones (por ejemplo, redimensionar las imágenes y normalizarlas)
transform = transforms.Compose([
    transforms.Resize((64, 64)),                                        # Cambia esto según el tamaño de tus imágenes
    transforms.ToTensor(),                                              # Convertir las imágenes a tensores
    transforms.Normalize(mean=[0.5], std=[0.5])     # Normalización
])


# Crear el dataset personalizado
dataset = CustomDataset(
    img1_dir='C:\Program Files\Blender Foundation\Blender 3.6\Img\Img_axis_x', 
    img2_dir='C:\Program Files\Blender Foundation\Blender 3.6\Img\Img_axis_y', 
    img3_dir='C:\Program Files\Blender Foundation\Blender 3.6\Img\Img_axis_z',
    coords_file='C:\Program Files\Blender Foundation\Blender 3.6\Img\coords.csv',
    transform=transform
)

# Dividir el dataset en entrenamiento y validación
train_size = int(0.7 * len(dataset))  # 70% para entrenamiento
val_size = int(0.15 * len(dataset))  # 15% para validación
test_size = len(dataset) - train_size - val_size  # 15% para test
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Crear DataLoaders para iterar sobre los datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# Inicializar modelo
model = CoordinatePredictor().to(device)

# Definir el optimizador y la función de pérdida
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Resumen del modelo
summary(model, input_size= [(1, 64, 64), (1, 64, 64), (1, 64, 64)])

# Supongamos que ya tenemos el modelo definido: model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Entrenamiento
num_epochs = 10
avg_val_loss = np.empty(num_epochs, dtype=np.float32)
avg_train_loss = np.empty(num_epochs, dtype=np.float32)
avg_val_loss = np.empty(num_epochs, dtype=np.float32)

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0

    for img1, img2, img3, target_coords in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(img1, img2, img3)
        loss = criterion(outputs, target_coords)

        # Backward pass y optimización
        loss.backward()
        optimizer.step()

    for img1, img2, img3, target_coords in train_loader:
        outputs = model(img1, img2, img3)
        loss = criterion(outputs, target_coords)        
        running_train_loss += loss.item()
    
    for img1, img2, img3, target_coords in val_loader:
        outputs = model(img1, img2, img3)
        loss = criterion(outputs, target_coords)        
        avg_val_loss[epoch] = loss.item()
        
    avg_train_loss[epoch] = running_train_loss / len(train_loader)
    print("Error medio de entrenamiento en iteración ", epoch, ": ", avg_train_loss[epoch], sep="")

    print("Error medio de validación en iteración ", epoch, ": ", avg_val_loss[epoch], sep="")

    # Condición de parada
    if epoch != 0 and avg_val_loss[epoch] > avg_val_loss[epoch-1]:
        print("Fin del entrenamiento por early stopping")
        break

print("Entrenamiento completado")
plt.plot(avg_train_loss, label="Training error")           # graficamos los MSE de cada ronda
plt.plot(avg_val_loss, label="Validation Error")
plt.title('Variación del error MSE a lo largo del entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Valor de pérdida')
plt.legend()
plt.show()

# Guardar el modelo
# torch.save(model.state_dict(), 'model.pth')

# # Cargar el modelo
# model = CoordinatePredictor()
# model.load_state_dict(torch.load('model.pth'), weights_only = True)
# model.eval()

# Evaluación en el conjunto de test
test_loss = 0.0
model.eval()
with torch.no_grad():
    for img1, img2, img3, target_coords in test_loader:
        outputs = model(img1, img2, img3)
        loss = criterion(outputs, target_coords)
        test_loss = loss.item()

print("Error medio de test: ", test_loss)

# Predicción de coordenadas para una muestra
sample_idx = 0
img1, img2, img3, target_coords = test_dataset[sample_idx]
img1 = img1.unsqueeze(0)
img2 = img2.unsqueeze(0)
img3 = img3.unsqueeze(0)
target_coords = target_coords.unsqueeze(0)

model.eval()
with torch.no_grad():
    predicted_coords = model(img1, img2, img3)

print("Coordenadas reales: ", target_coords)
print("Coordenadas predichas: ", predicted_coords)