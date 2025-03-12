import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
#from IPython.display import clear_output
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#import mplcursors

## Set Seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
set_seed(42)

st.session_state.entrainement_modele=True

# Custom callback to plot training and validation loss during training

# Custom Training Monitor
class TrainingMonitorNotebook:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    def update_plot(self, epoch, train_loss, val_loss):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

        #clear_output(wait=True)

        plt.figure(figsize=(5, 3))  # Adjust size to occupy half of the window
        train_line, = plt.plot(self.train_loss, label="Training Loss", color='blue')
        val_line, = plt.plot(self.val_loss, label="Validation Loss", color='orange')
        plt.xlabel("Epochs", fontsize=6)
        plt.ylabel("Loss", fontsize=6)
        plt.title("Training and Validation Loss Over Epochs", fontsize=6)
        plt.legend(fontsize=6)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.grid(True)

        #mplcursors.cursor([train_line, val_line], hover=True)
        plt.show()

# Instantiate the monitor
training_monitor_notebook = TrainingMonitorNotebook()

# BiLSTM Model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True, num_layers=2)
        self.bilstm = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True, num_layers=2)
        self.bilstm = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True, num_layers=2)
        self.fc2 = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional LSTM

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = x.view(-1, 2, 512)  # Adjust based on input
        x, _ = self.bilstm(x)
        x = x[:, -1, :]
        return self.fc2(x)

# Model Parameters
input_size = 60
hidden_size = 10
output_size = 5  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMModel(input_size, hidden_size, output_size).to(device)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



def entrainer_le_modèle(dossier_donnees):
    if len(os.listdir(dossier_donnees)) > 0:
        # Liste tous les fichiers dans le dossier
        fichiers = [f for f in os.listdir(dossier_donnees) if os.path.isfile(os.path.join(dossier_donnees, f))]
        st.write("Entraînement avec les fichiers suivants en cours ...", )
        # Affiche la liste des fichiers
        for fichier in fichiers:
            st.write(fichier)
        
        # Load the CSV files
        x_train = pd.read_csv(dossier_donnees + "x_train_s13_o60_p5.csv", header=None)
        y_train = pd.read_csv(dossier_donnees + "y_train_s13_o60_p5.csv", header=None)
        x_valid = pd.read_csv(dossier_donnees + "x_valid_s65_o60_p5.csv", header=None)
        y_valid = pd.read_csv(dossier_donnees + "y_valid_s65_o60_p5.csv", header=None)

        min_y_val = y_valid.values.min()
        max_y_val = y_valid.values.max()

        # StandardScaler rescaling
        # For x_train and x_valid (60 features)
        x_scaler = StandardScaler()
        x_train_StandardScaler = x_scaler.fit_transform(x_train)
        x_valid_StandardScaler = x_scaler.transform(x_valid)

        # For y_train and y_valid (5 features)
        y_scaler = StandardScaler()
        y_train_StandardScaler = y_scaler.fit_transform(y_train)
        y_valid_StandardScaler = y_scaler.transform(y_valid)

        x_train_standardized = x_train_StandardScaler
        x_valid_standardized = x_valid_StandardScaler
        y_train_standardized = y_train_StandardScaler
        y_valid_standardized = y_valid_StandardScaler

        # Convert datasets to PyTorch tensors
        x_train_tensor = torch.tensor(x_train_standardized, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train_standardized, dtype=torch.float32).to(device)
        x_valid_tensor = torch.tensor(x_valid_standardized, dtype=torch.float32).to(device)
        y_valid_tensor = torch.tensor(y_valid_standardized, dtype=torch.float32).to(device)

        #st.write(y_valid_standardized)



        # Training Parameters
        epochs = 10
        batch_size = 2048

        # Create a Streamlit placeholder for the plot
        plot_placeholder = st.empty()

        # Training Loop with Time Series Handling (No Shuffling)
        for epoch in range(epochs):
            model.train()

            epoch_loss = 0
            num_batches = 0

            for i in range(0, x_train_tensor.size(0), batch_size):
                batch_x = x_train_tensor[i:i + batch_size]
                batch_y = y_train_tensor[i:i + batch_size]

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1  # Count actual batches

            avg_train_loss = epoch_loss / num_batches  # Fix loss calculation

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(x_valid_tensor)
                val_loss = criterion(val_outputs, y_valid_tensor).item()

            #st.write(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")

            # Mettre à jour en live le graphique des métriques
            training_monitor_notebook.update_plot(epoch, avg_train_loss, val_loss)

            # Afficher le graph des métriques
            with plot_placeholder.container():
                plt.figure(figsize=(4, 2))
                plt.plot(training_monitor_notebook.train_loss, label="Training Loss", color='blue')
                plt.plot(training_monitor_notebook.val_loss, label="Validation Loss", color='orange')
                plt.xlabel("Epochs", fontsize=5)
                plt.ylabel("Loss", fontsize=5)
                plt.title("Training and Validation Loss par Epoch", fontsize=5)
                plt.legend(fontsize=5)
                plt.xticks(fontsize=5)
                plt.yticks(fontsize=5)
                plt.grid(True)
                st.pyplot(plt)

        # Validation KPI
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_valid_tensor)
            val_loss = criterion(val_outputs, y_valid_tensor)
            print(f"Validation Loss: {val_loss.item()}")

        # Move val_outputs to CPU, then convert to NumPy and perform inverse scaling
        y_pred = y_scaler.inverse_transform(val_outputs.cpu().numpy()).astype(int)
        #y_pred = scaler.inverse_transform(val_outputs.cpu().numpy()).astype(int)

        print("y_pred.shape: ", y_pred.shape, "\n")

        print(y_pred[:4])

        print("\ny_pred.min(): ",y_pred.min())

        # Replace all negative values with zero
        y_pred = np.where(y_pred < 0, 0, y_pred)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        print(f'Validation RMSE: {rmse}')

        # Calculate nRMSE
        nrmse_KPI = np.sqrt(mean_squared_error(y_valid, y_pred))/(max_y_val-min_y_val)
        st.write(f'VMSE: {nrmse_KPI}')
        st.session_state.nrmse_value = nrmse_KPI

    else:
        st.write("Aucun fichier validé disponible. Procédez au dépot et à la validation des données avant.")

    return 1