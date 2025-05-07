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
from pytorch3d.loss import chamfer_distance
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

data_type = "ISAR"

if data_type == "ISAR" or data_type == "npy":
    pass
else:
    raise ValueError("data_type must be one of the following cases: [\"ISAR\"]")

print(f"\nUsing {device} device\n") 

class CustomDataset_npy(Dataset):
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
        self.file_names = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
        
        # Sort files based on the sample id which is the number right after "sample_"
        # For example, "sample_5_322.806383_T.npy" will be split into
        # ['sample', '5', '322.806383', 'T.npy'].
        self.file_names.sort(key=lambda x: int(x.split('_')[1]))
        
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

class CustomDataset_ISAR(Dataset):

    def __init__(self, img1_dir, img2_dir, img3_dir, coords_file, transform=None):
        # Directorios de imágenes y archivo CSV de coordenadas
        self.img1_dir = img1_dir
        self.img2_dir = img2_dir
        self.img3_dir = img3_dir
        self.coords = np.load(coords_file)[:, 1:]
        self.transform = transform
        self.dist_max = self.dist_max_calc()

    def dist_max_calc(self):
        sample_coords = self.coords[1].reshape(int(self.coords.shape[1]/3),3)
        dist_max = 0.0
        for i in range(len(sample_coords)):
            if sum(sample_coords[i] ** 2) > dist_max:
                dist_max = sum(sample_coords[i] ** 2)
            dist_max = np.sqrt(dist_max)
        return dist_max
    
    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):

        # Cargar las tres imágenes
        img1_path = os.path.join(self.img1_dir, f"ISAR_x/isar_{idx+1}_1.png") #sample_{idx+1}_x.png
        img2_path = os.path.join(self.img2_dir, f"ISAR_y/isar_{idx+1}_2.png")
        img3_path = os.path.join(self.img3_dir, f"ISAR_z/isar_{idx+1}_3.png")
        img1 = Image.open(img1_path).convert("L")
        img2 = Image.open(img2_path).convert("L")
        img3 = Image.open(img3_path).convert("L")

        # Aplicar transformaciones si las hay
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        # Obtener el vector de coordenadas de salida
        coords = torch.tensor(self.coords[idx,:], dtype=torch.float32)

        return img1.to(device), img2.to(device), img3.to(device), coords.to(device)

class CNNModule_npy(nn.Module):
    """Convolutional module to extract features from a 2-channel input.

    This network processes an input tensor with shape (batch_size, 2, height, width)
    and returns a flattened feature vector.
    """

# La red LSTM podria usarse para relacionar columnas o filas del npy entre sí

    def __init__(self):
        super(CNNModule_npy, self).__init__()
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
    
class CNNModule_ISAR(nn.Module):

    def __init__(self):
        super(CNNModule_ISAR, self).__init__()
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
        self.cnn = CNNModule_npy()

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

class CoordinatePredictor(nn.Module):

    def __init__(self,coords_width):
        super(CoordinatePredictor, self).__init__()
        # CNN compartida para las tres imágenes
        self.cnn = CNNModule_ISAR()

        # Fully connected layers
        self.fc1 = nn.LazyLinear(4096)  # Ajustar el tamaño si las imágenes son diferentes
        self.fc2 = nn.LazyLinear(4096)
        #self.Dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.LazyLinear(coords_width)  # 3*N salidas, una por coordenada (x, y, z) de cada uno de los N vertices

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
        output_coords = self.fc3(x)

        return output_coords
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
transform_ISAR = transforms.Compose([
    transforms.Resize((128, 128)),                  # Cambia esto según el tamaño de tus imágenes
    transforms.ToTensor(),                          # Convertir las imágenes a tensores
    transforms.Normalize(mean=[0.5], std=[0.5])     # Normalización
])

# Crear el dataset personalizado
if data_type == "npy":
    dataset = CustomDataset_npy(
        npy_dir=f"/home/newfasant/N101-IA/Datasets/Reorganized/Classification_300",
        labels_file=f"/home/newfasant/N101-IA/Datasets/Reorganized/labels_vector_300.npy"
    )
    # Inicializar modelo
    model = Geometry_Predictor(len(np.unique(dataset.labels_vector))).to(device)

    # Definir el optimizador y la función de pérdida
    optimizer = optim.SGD(params=model.parameters(), lr=0.01)
    criterion_CEL = nn.CrossEntropyLoss()
    
elif data_type == "ISAR":
    dataset = CustomDataset_ISAR(
    img1_dir=f"./Img/{data_type}",
    img2_dir=f"./Img/{data_type}",
    img3_dir=f"./Img/{data_type}",
    coords_file=f"./Img/coords.npy",
    transform=transform_ISAR
    )
    # Inicializar modelo
    model = CoordinatePredictor(dataset.coords.shape[1]).to(device)

    # Definir el optimizador y la función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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

# Early Stopping Instance
early_stopping = EarlyStopping(patience=100)


# Entrenamiento
num_epochs = 250
num_train_batches=len(train_loader)
avg_val_loss = np.empty(num_epochs, dtype=np.float32)
avg_train_loss = np.empty(num_epochs, dtype=np.float32)
avg_val_loss = np.empty(num_epochs, dtype=np.float32)

for epoch in range(num_epochs):
    
    start_time = time.time()
    running_train_loss = 0.0
    
    model.train()
    if data_type == "npy":
        for npy, target_label in train_loader:
            optimizer.zero_grad()

            # Forward pass
            output_logits = model(npy) # This output gives the logits, not the probabilities
            pred_probs= F.softmax(output_logits, dim=1) # This converts the logits to probabilities

            loss = criterion_CEL(output_logits, target_label.long())
            running_train_loss += loss.item()

            # Backward pass y optimización
            loss.backward()
            optimizer.step()
    elif data_type == "ISAR":
        for img1, img2, img3, target_coords in train_loader:
            optimizer.zero_grad()

            outputs = model(img1, img2, img3)

            outputs = outputs.view(-1, int(dataset.coords.shape[1]/3), 3)
            target_coords = target_coords.view(-1, int(dataset.coords.shape[1]/3), 3)
            lossChamfer, _ = chamfer_distance(outputs, target_coords)

            lossChamfer.backward()
            optimizer.step()

            running_train_loss += lossChamfer.item()
    
    model.eval()
    with torch.no_grad():
        if data_type == "npy":
            for npy, target_label in val_loader:
                
                output_logits = model(npy)
                pred_probs = F.softmax(output_logits, dim=1)

                loss = criterion_CEL(output_logits, target_label.long())
                avg_val_loss[epoch] = loss.item()
        
        elif data_type == "ISAR":
            for img1, img2, img3, target_coords in val_loader:
                outputs = model(img1, img2, img3)

                outputs = outputs.view(-1, int(dataset.coords.shape[1]/3), 3)  
                target_coords = target_coords.view(-1, int(dataset.coords.shape[1]/3), 3)
                lossChamfer, _ = chamfer_distance(outputs, target_coords)

                avg_val_loss[epoch] = lossChamfer.item()
    # Check Early Stopping
    if early_stopping(avg_val_loss[epoch]):
        break  # Stop training
    
    end_time = time.time()
    epoch_duration = end_time - start_time
    avg_train_loss[epoch] = running_train_loss / num_train_batches
    if data_type == "npy":
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_duration:.2f} seconds with a training error of {avg_train_loss[epoch]:.4f} a validation loss of {avg_val_loss[epoch]:.4f}")
    elif data_type == "ISAR":
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_duration:.2f} seconds with a training error of {avg_train_loss[epoch]:.4f}, a validation loss of {avg_val_loss[epoch]:.4f} and a relative training error of {avg_train_loss[epoch]/dataset.dist_max:.4f}")

print("Entrenamiento completado")

# Guardar el modelo
torch.save(model.state_dict(), f"{os.path.basename(os.getcwd())}_{data_type}_{len(dataset)}samples_{num_epochs}ep_{train_batch_size}bs.pth")

# Evaluación en el conjunto de test
test_loss = 0.0
model.eval()
with torch.no_grad():
    if data_type == "npy":
        for npy, target_label in test_loader:
            
            output_logits = model(npy)
            pred_probs = F.softmax(output_logits, dim=1)
            loss = criterion_CEL(output_logits, target_label.long())
            
            test_loss = loss.item()
    elif data_type == "ISAR":
        for img1, img2, img3, target_coords in test_loader:
            outputs = model(img1, img2, img3)

            outputs = outputs.view(-1, int(dataset.coords.shape[1]/3), 3)  
            target_coords = target_coords.view(-1, int(dataset.coords.shape[1]/3), 3)
            lossChamfer, _ = chamfer_distance(outputs, target_coords)
            
            test_loss = lossChamfer.item()
print("Error medio de test: ", test_loss)

