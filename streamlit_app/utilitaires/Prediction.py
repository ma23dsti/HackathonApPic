import joblib
import json
import numpy as np
import random
import torch
import torch.nn as nn

# Modèle
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 1000)
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(500, hidden_size, bidirectional=True, batch_first=True, num_layers=2)
        self.bilstm = nn.LSTM(500, hidden_size, bidirectional=True, batch_first=True, num_layers=2)
        self.bilstm = nn.LSTM(500, hidden_size, bidirectional=True, batch_first=True, num_layers=2)
        self.fc2 = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional LSTM

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = x.view(-1, 2, 500)  # Adjust based on input
        x, _ = self.bilstm(x)
        x = x[:, -1, :]
        return self.fc2(x)

def predire_le_traffic(donnees_observees):

    # Charger le modèle et ses hyperparamètres.

    with open("streamlit_app/static/modeles/modele_par_defaut/modele_parametres.json", "r") as f:
        params = json.load(f)

    loaded_input_size = params["input_size"]
    loaded_hidden_size = params["hidden_size"]
    loaded_output_size = params["output_size"]

    # Reconstruire le modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = BiLSTMModel(loaded_input_size, loaded_hidden_size, loaded_output_size).to(device)

    # Charger les poids du modèle
    loaded_model.load_state_dict(torch.load("streamlit_app/static/modeles/modele_par_defaut/modele.pth", map_location=device))
    loaded_model.eval()  # Set model to evaluation mode

    # Charger les transformations des données
    loaded_x_scaler = joblib.load("streamlit_app/static/modeles/modele_par_defaut/x_scaler.pkl")
    loaded_y_scaler = joblib.load("streamlit_app/static/modeles/modele_par_defaut/y_scaler.pkl")

    # Prédire à partir de l'historique fourni

    x_valid_standardized = loaded_x_scaler.transform(donnees_observees)
    x_valid_tensor = torch.tensor(x_valid_standardized, dtype=torch.float32).to(device)

    # Perform inference to forecast the next x steps
    pred_model_loaded = loaded_model(x_valid_tensor)

    # Move to CPU before converting to NumPy
    pred_model_loaded = pred_model_loaded.cpu().detach().numpy()

    # Use the scaler to inverse transform
    pred_model_loaded = loaded_y_scaler.inverse_transform(pred_model_loaded).astype(int)

    # Replace all negative values with zero
    pred_model_loaded = np.where(pred_model_loaded < 0, 0, pred_model_loaded)*random.uniform(0.5, 1.5)

    # Dummy different models predictions
    # Generate random multipliers for each element
    random_multipliers = np.random.uniform(0.5, 1.5, size=pred_model_loaded.shape)
    pred_model_loaded = pred_model_loaded * random_multipliers

    print("Prediction terminée")

    return(pred_model_loaded)