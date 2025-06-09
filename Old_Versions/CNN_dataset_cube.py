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
import logging
# from torchsummary import summary
from pytorch3d.loss import chamfer_distance
import csv
import time
import argparse
from datetime import datetime

# Obtener la fecha y hora actual
actual_time = datetime.now()

logging.basicConfig(
    filename=f'/home/newfasant/N101-IA/CNN/Logs/Day_{actual_time.day}_{actual_time.month}_{actual_time.year}_Time_{actual_time.hour:02d}_{actual_time.minute:02d}_NN.log',  # Name of the log file
    level=logging.INFO,  # Logs level (INFO, DEBUG, ERROR, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

parser = argparse.ArgumentParser(description='Procesar argumentos para enviarlos a los distintos programas necesarios.')
parser.add_argument('-i', '--input_path', type=str, help='Ruta de la carpeta en la que se encuentra el dataset (terminada en una carpeta formato Classification_660_0_64_f_64_d).')
parser.add_argument('-t', '--data_type', type=str, default='ISAR', choices=['ISAR', 'npy'], help='Tipo de dato que se desea usar para entrenar la red.')
parser.add_argument('-u', '--use_case', type=str, default='Classification', choices=['Classification', 'Regression'], help='Tipo de uso que deseado  a la red')

args = parser.parse_args()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
         )

torch.manual_seed(101)
torch.cuda.manual_seed(101)

print(f"\nUsing {device} device, {args.data_type} data type and {args.use_case} use case.\n")
logging.info(f"\nUsing {device} device, {args.data_type} data type and {args.use_case} use case.\n")

class CustomDataset(Dataset):
    """Custom dataset for loading data files and their corresponding labels.

    Initializes the CustomDataset instance.

    Args:
        npy_dir (str): Directory containing the data files.
        labels_file (str): File path to the file containing labels.
        transform_ISAR (callable, optional): Optional transform to be applied on a sample for ISAR data type.
        transform_npy (callable, optional): Optional transform to be applied on a sample for npy data type.
    """

    def __init__(self, dir, transform_ISAR=None, transform_npy=None):

        self.dir = dir

        if args.data_type == 'npy':
            # List all .npy files in the directory.
            self.file_names = [f for f in os.listdir(dir) if f.endswith('.npy') and f.startswith('sample_')]
            self.transform = transform_npy
        elif args.data_type == 'ISAR':
            # List all .npy files in the directory.
            self.file_names = [f for f in os.listdir(dir) if f.endswith('.png') and f.startswith('sample_')]
            self.transform = transform_ISAR
            
        if args.use_case == "Classification":
            _, _, result = dir.split('/')[-1].partition('_')  # divide en 3 partes: antes, separador, después
            result = f"_{result}"
            # Load labels vector from the labels file.
            self.labels_vector = np.load(dir + '/labels_vector' + result + '.npy')
        elif args.use_case == "Regression":
            self.coords = np.load(dir + "/coords.npy")[:, 1:]
            self.dist_max = self.dist_max_calc()
            
        # Sort files based on the sample id which is the number right after "sample_". For example, "sample_5.npy" will be split into ['sample', '5', '.npy'].
        self.file_names.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            
    def __len__(self):
        """Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.file_names)
    
    def __getitem__(self, idx):

        # Construct the full file path for the sample.
        file_path = os.path.join(self.dir, self.file_names[idx])
        
        if args.data_type == 'npy':
            # Load the numpy array with shape (n x m x 2).
            data = np.load(file_path)
        
            # Convert the numpy array to a PyTorch tensor.
            data = torch.tensor(data, dtype=torch.float32).to(device)
        elif args.data_type == 'ISAR':
            data = Image.open(file_path).convert("L")

            # Aplicar transformaciones si las hay
            if self.transform:
                data = self.transform(data).to(device)
            else:
                data.to(device)

        if args.use_case == "Classification":
            # Load labels vector from the labels file.
            target = torch.tensor(self.labels_vector[idx], dtype=torch.float32).to(device)
        elif args.use_case == "Regression":
            target = torch.tensor(self.coords[idx,:], dtype=torch.float32).to(device)
        
        return data, target
    
    def dist_max_calc(self):
        sample_coords = self.coords[1].reshape(int(self.coords.shape[1]/3),3)
        dist_max = 0.0
        for i in range(len(sample_coords)):
            if sum(sample_coords[i] ** 2) > dist_max:
                dist_max = sum(sample_coords[i] ** 2)
            dist_max = np.sqrt(dist_max)
        return dist_max

class CNNModule(nn.Module):
    """Convolutional module to extract features from a 2-channel input.

    This network processes an input tensor with shape (batch_size, 2, height, width)
    and returns a flattened feature vector.
    """
    # La red LSTM podria usarse para relacionar columnas o filas del npy entre sí

    def __init__(self):
        super(CNNModule, self).__init__()
        
        # Capas convolucionales
        if args.data_type == 'npy':
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

        elif args.data_type == 'ISAR':
            self.conv1 = nn.LazyConv2d(out_channels=96, kernel_size=22, stride=2, padding=1)
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
    
class GeometryPredictor(nn.Module):

    def __init__(self, num_labels=None, coords_width=None):

        super(GeometryPredictor, self).__init__()
        
        self.cnn = CNNModule()

        # Fully connected layers
        self.fc1 = nn.LazyLinear(4096)  # Adjust size if needed.
        self.fc2 = nn.LazyLinear(4096)
        
        if args.use_case == "Classification":
            self.fc3 = nn.LazyLinear(num_labels)  # Number of outputs equal to num_labels.
        if args.use_case == "Regression":
            self.fc3 = nn.LazyLinear(coords_width)  # 3*N salidas, una por coordenada (x, y, z) de cada uno de los N vertices

    def forward(self, x):

        # Process the input through the CNN to extract features.
        features = self.cnn(x)

        # Pass the features through fully connected layers.
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        return output
    
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
                print(f"\nEarly stopping triggered after {self.patience} epochs.\n")
                logging.info(f"\nEarly stopping triggered after {self.patience} epochs.\n")

                return True
        return False

transform_I = transforms.Compose([
    transforms.Resize((128, 128)),                                      # Cambiar según el tamaño de las imágenes
    transforms.ToTensor(),                                              # Convertir las imágenes a tensores
    transforms.Normalize(mean=[0.5], std=[0.5])     # Normalización
])

dataset = CustomDataset(dir = args.input_path, transform_ISAR=transform_I)


if args.use_case == "Classification":
    # Inicializar modelo
    model = GeometryPredictor(num_labels=len(np.unique(dataset.labels_vector))).to(device)

    # Definir el optimizador y la función de pérdida
    optimizer = optim.SGD(params=model.parameters(), lr=0.01)
    criterion_CEL = nn.CrossEntropyLoss()
    
elif args.use_case == "Regression":
    # Inicializar modelo
    model = GeometryPredictor(coords_width=dataset.coords.shape[1], num_labels=len(np.unique(dataset.labels_vector))).to(device)

    # Definir el optimizador y la función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dividir el dataset en entrenamiento y validación
train_size = int(0.7 * len(dataset))  # 70% para entrenamiento
val_size = int(0.15 * len(dataset))  # 15% para validación
test_size = len(dataset) - train_size - val_size  # 15% para test

generator = torch.Generator().manual_seed(101)

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

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
    if args.use_case == "Classification":
        for data, target_label in train_loader:
            optimizer.zero_grad()

            # Forward pass
            output_logits = model(data) # This output gives the logits, not the probabilities
            pred_probs= F.softmax(output_logits, dim=1) # This converts the logits to probabilities

            loss = criterion_CEL(output_logits, target_label.long())
            running_train_loss += loss.item()

            # Backward pass y optimización
            loss.backward()
            optimizer.step()
    
    elif args.use_case == "Regression":
        for data, target_coords in train_loader:
            optimizer.zero_grad()

            outputs = model(data)
            outputs = outputs.view(-1, int(dataset.coords.shape[1]/3), 3)
            target_coords = target_coords.view(-1, int(dataset.coords.shape[1]/3), 3)
            
            lossChamfer, _ = chamfer_distance(outputs, target_coords)
            running_train_loss += lossChamfer.item()

            # Backward pass y optimización
            lossChamfer.backward()
            optimizer.step()

    
    model.eval()
    with torch.no_grad():
        if args.use_case == "Classification":
            for data, target_label in val_loader:
                
                output_logits = model(data)
                pred_probs = F.softmax(output_logits, dim=1)

                loss = criterion_CEL(output_logits, target_label.long())
                avg_val_loss[epoch] = loss.item()
        
        elif args.use_case == "Regression":
            for data, target_coords in val_loader:
                outputs = model(data)

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

    if args.use_case == "Classification":
        print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_duration:.2f} seconds with a training error of {avg_train_loss[epoch]:.4f} a validation loss of {avg_val_loss[epoch]:.4f}\n")
        logging.info(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_duration:.2f} seconds with a training error of {avg_train_loss[epoch]:.4f} a validation loss of {avg_val_loss[epoch]:.4f}\n")
    elif args.use_case == "Regression":
        print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_duration:.2f} seconds with a training error of {avg_train_loss[epoch]:.4f}, a validation loss of {avg_val_loss[epoch]:.4f} and a relative training error of {avg_train_loss[epoch]/dataset.dist_max:.4f}\n")
        logging.info(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_duration:.2f} seconds with a training error of {avg_train_loss[epoch]:.4f}, a validation loss of {avg_val_loss[epoch]:.4f} and a relative training error of {avg_train_loss[epoch]/dataset.dist_max:.4f}\n")

print("Entrenamiento completado\n")
logging.info("Entrenamiento completado\n")

# Guardar el modelo
torch.save(model.state_dict(), f"{os.getcwd()}/Models/{args.use_case}_{args.data_type}_{len(dataset)}samples_{num_epochs}ep_{train_batch_size}bs.pth")
 
print("Model saved\n")
logging.info("Model saved\n")

# Evaluación en el conjunto de test
test_loss = 0.0
model.eval()
with torch.no_grad():
    if args.use_case == "Classification":
        for npy, target_label in test_loader:
            
            output_logits = model(npy)
            pred_probs = F.softmax(output_logits, dim=1)
            loss = criterion_CEL(output_logits, target_label.long())
            
            test_loss = loss.item()

    elif args.use_case == "Regression":
        for img, target_coords in test_loader:
            
            outputs = model(img)

            outputs = outputs.view(-1, int(dataset.coords.shape[1]/3), 3)  
            target_coords = target_coords.view(-1, int(dataset.coords.shape[1]/3), 3)
            lossChamfer, _ = chamfer_distance(outputs, target_coords)
            
            test_loss = lossChamfer.item()

print("Error medio de test: ", test_loss)
logging.info("Error medio de test: ", test_loss)

print("Fin de la ejecución")
logging.info("Fin de la ejecución")
