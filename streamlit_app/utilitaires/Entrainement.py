import joblib
import json
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st
import numpy as np
import random
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
#from IPython.display import clear_output
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

# Custom Training Monitor callback to plot training and validation loss during training
class TrainingMonitorNotebook:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.cumulative_epochs = []  # Store cumulative epoch numbers
        self.session_start_indices = []  # Track where each session starts

    def update_plot(self, epoch, train_loss, val_loss):
        # Append the current epoch number and losses
        if len(self.cumulative_epochs) > 0 and epoch == 0:
            # Add NaN to break the line between sessions
            self.cumulative_epochs.append(np.nan)
            self.train_loss.append(np.nan)
            self.val_loss.append(np.nan)
            self.session_start_indices.append(len(self.cumulative_epochs))  # Mark the start of a new session

        self.cumulative_epochs.append(len(self.cumulative_epochs) + 1)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

    def generate_labels(self):
        # Determine spacing based on cumulative epoch count
        total_epochs = len(self.cumulative_epochs)
        if total_epochs > 500:
            spacing = 50
        elif total_epochs > 400:
            spacing = 40
        elif total_epochs > 300:
            spacing = 30
        elif total_epochs > 200:
            spacing = 20
        elif total_epochs > 100:
            spacing = 10
        elif total_epochs > 25:
            spacing = 5
        else:
            spacing = 1

        # Generate non-cumulative labels for each session
        labels = [""] * len(self.cumulative_epochs)
        for i, start_idx in enumerate(self.session_start_indices + [len(self.cumulative_epochs)]):
            if i == 0:
                session_start = 0
            else:
                session_start = self.session_start_indices[i - 1]

            for j in range(session_start, start_idx):
                non_cumulative_epoch = j - session_start + 1
                if non_cumulative_epoch % spacing == 0:
                    labels[j] = str(non_cumulative_epoch)
        return labels

    def reset_plot(self):
        self.train_loss = []
        self.val_loss = []
        self.cumulative_epochs = []
        self.session_start_indices = []

# Instantiate the monitor
training_monitor_notebook = TrainingMonitorNotebook()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def entrainer_le_modèle(dossier_donnees):
    st.write(st.session_state.on_homepage)
    # Reset the training plot when navigating to the homepage
    if st.session_state.on_homepage:
        training_monitor_notebook.reset_plot()
        st.session_state["on_homepage"] = False  # Reset the flag after clearing the plot

    # Tailles nécessaires pour l'entrainement'
    horizon = st.session_state.horizon_predictions
    taille_fenetre_observee = st.session_state.taille_fenetre_observee
    sliding_window_train = st.session_state.sliding_window_train
    sliding_window_valid = st.session_state.sliding_window_valid

    # Dossier du modèle par défaut
    dossier_modele_par_defaut = f"streamlit_app/static/modeles/modele_par_defaut/modele_par_defaut_restreint_o{taille_fenetre_observee}_p{horizon}/"
    # Récupérer les fichiers du modèle par défaut à utiliser pour l'entrainement
    fichiers_modele = ["modele.pth", "modele_parametres.json", "x_scaler.pkl", "y_scaler.pkl"]
    dossier_modele_courant = "streamlit_app/static/modeles/modele_courant/"
    # Créer le dossier du modèle courant s'il n'existe pas.
    os.makedirs(dossier_modele_courant, exist_ok=True)

    # Copier chaque fichier dans le dossier du modèle courant
    for fichier in fichiers_modele:
        chemin_source = os.path.join(dossier_modele_par_defaut, fichier)
        chemin_destination = os.path.join(dossier_modele_courant, fichier)

        # Vérifier si le fichier existe dans le dossier de destination et le supprimer
        if os.path.exists(chemin_destination):
            os.remove(chemin_destination)
            print(f"Supprimé : {chemin_destination}")

        # Vérifier si le fichier source existe avant la copie
        if os.path.exists(chemin_source):  
            shutil.copy(chemin_source, chemin_destination)
            print(f"Copié : {fichier} → {chemin_destination}")
        else:
            print(f"Fichier introuvable : {chemin_source}")

        # Charger le nRMSE de référence à partir du fichier JSON et le sauvegarder dans st.session_state.baseline_nrmse
        try:
            with open(dossier_modele_courant + "modele_parametres.json", "r") as f:
                params = json.load(f)
                st.session_state.baseline_nrmse = params.get("kpi", {}).get("nrmse", None)
        except FileNotFoundError:
            st.session_state.baseline_nrmse = 0.15  # Valeur par défaut si le fichier n'est pas trouvé
            # Nombre maximum de tentatives
            max_retries = 3
            retry_interval = 2  # secondes entre chaque tentative
            # Tentatives de lecture après la copie du fichier
            for attempt in range(max_retries):
                if os.path.exists(dossier_modele_courant + "modele_parametres.json"):  # Vérifie si le fichier existe
                    st.write("Le fichier modele_parametres.json a été trouvé !")
                    # Code pour charger ou traiter le fichier ici
                    break
                else:
                    # Si le fichier n'est pas trouvé, afficher un message et attendre avant de réessayer
                    st.write(f"Tentative {attempt + 1} : Le fichier modele_parametres.json est introuvable.")
                    if attempt < max_retries - 1:  # Si ce n'est pas la dernière tentative
                        time.sleep(retry_interval)  # Attendre avant la prochaine tentative
                    else:
                        st.write("Le fichier n'a pas pu être trouvé après plusieurs tentatives.")
        except json.JSONDecodeError:
            st.session_state.baseline_nrmse = 0.15 # Valeur par défaut si le fichier n'est pas trouvé
            st.write("Erreur lors du chargement du fichier modele_parametres.json.")

    # Architecture du modèle par défaut utilisé : BiLSTM
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

    # Charger le modèle préentrainé et ses hyperparamètres.
    with open(dossier_modele_courant + "modele_parametres.json", "r") as f:
        params = json.load(f)

    loaded_input_size = params["input_size"]
    loaded_hidden_size = params["hidden_size"]
    loaded_output_size = params["output_size"]

    # Reconstruire l'architecture du modèle à utiliser
    loaded_model = BiLSTMModel(loaded_input_size, loaded_hidden_size, loaded_output_size).to(device)
    # Charger les poids du modèle préentrainé
    loaded_model.load_state_dict(torch.load(dossier_modele_courant + "modele.pth", map_location=device))
    # Affecter le modèle par défaut préentrainé pour utilisation
    model = loaded_model

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    horizon = st.session_state.horizon_predictions
    taille_fenetre_observee = st.session_state.taille_fenetre_observee
    sliding_window_train = st.session_state.sliding_window_train
    sliding_window_valid = st.session_state.sliding_window_valid

    if len(os.listdir(dossier_donnees)) > 0:
        # Liste des fichiers nécessaires à l'entrainement
        nom_fichier_x_train = f"x_train_s{sliding_window_train}_o{taille_fenetre_observee}_p{horizon}.csv"
        nom_fichier_y_train = f"y_train_s{sliding_window_train}_o{taille_fenetre_observee}_p{horizon}.csv"
        nom_fichier_x_valid = f"x_valid_s{sliding_window_valid}_o{taille_fenetre_observee}_p{horizon}.csv"
        nom_fichier_y_valid = f"y_valid_s{sliding_window_valid}_o{taille_fenetre_observee}_p{horizon}.csv"
        noms_fichiers_entrainements = [nom_fichier_x_train, nom_fichier_y_train, nom_fichier_x_valid, nom_fichier_y_valid]

        # Liste de tous les fichiers dans le dossier
        fichiers = [f for f in os.listdir(dossier_donnees) if os.path.isfile(os.path.join(dossier_donnees, f))]
        # Vérifier que tous les fichiers nécessaires à l'entrainement sont présents et les afficher
        fichiers_manquants = [f for f in noms_fichiers_entrainements if f not in fichiers]
        if fichiers_manquants:
            st.write(f"Les fichiers suivants sont manquants : {fichiers_manquants}")
        
        # Load the CSV files
        x_train = pd.read_csv(dossier_donnees + nom_fichier_x_train, header=None)
        y_train = pd.read_csv(dossier_donnees + nom_fichier_y_train, header=None)
        x_valid = pd.read_csv(dossier_donnees + nom_fichier_x_valid, header=None)
        y_valid = pd.read_csv(dossier_donnees + nom_fichier_y_valid, header=None)

        min_y_val = y_valid.values.min()
        max_y_val = y_valid.values.max()

        # StandardScaler rescaling
        # For x_train and x_valid (st.session_state.taille_fenetre_observee features)
        x_scaler = StandardScaler()
        x_train_StandardScaler = x_scaler.fit_transform(x_train)
        x_valid_StandardScaler = x_scaler.transform(x_valid)

        # For y_train and y_valid (st.session_state.horizon_predictions features)
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
        epochs = 5
        batch_size = 1024

        # Create a Streamlit placeholder for the plot
        plot_placeholder = st.empty()

        # Training Loop with Time Series Handling (No Shuffling)
        progress_bar = st.progress(0)  # Initialize the progress bar
        progress_text = st.empty()  # Placeholder for progress text
        kpi_placeholder = st.empty()  # Placeholder for KPI display

        # Display files used for training outside the loop
        progress_text.write(
            f"Fichiers utilisés: {nom_fichier_x_train}, {nom_fichier_y_train}, {nom_fichier_x_valid}, {nom_fichier_y_valid}"
        )

        for epoch in range(epochs):
            # Shuffle les données d'entrainement après chaque epoch.
            # Le shuffling est effectué entre portions de série temporelle pour un meilleur apprentissage,
            # et non entre toutes les données brutes unitaires pour ne pas supprimer les liens de la série temporelle.
            indices = np.arange(len(x_train_tensor))
            np.random.shuffle(indices)
            x_train_tensor = x_train_tensor[indices]
            y_train_tensor = y_train_tensor[indices]

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

            # Update the progress bar
            progress_bar.progress((epoch + 1) / epochs)

            # Clear the previous KPI line and display the new one
            kpi_placeholder.empty()
            kpi_placeholder.write(f"Epoch {epoch + 1}/{epochs} | Training Loss: {avg_train_loss:.4f} | Validation Loss: {val_loss:.4f}")

            # Mettre à jour en live le graphique des métriques
            training_monitor_notebook.update_plot(epoch, avg_train_loss, val_loss)

        # Afficher le graph des métriques with filtered x-axis labels
        with plot_placeholder.container():
            plt.figure(figsize=(4, 2))
            plt.plot(
                training_monitor_notebook.cumulative_epochs,
                training_monitor_notebook.train_loss,
                label="Training Loss",
                color='blue'
            )
            plt.plot(
                training_monitor_notebook.cumulative_epochs,
                training_monitor_notebook.val_loss,
                label="Validation Loss",
                color='orange'
            )
            plt.xlabel("Epochs", fontsize=5)
            plt.ylabel("Loss", fontsize=5)
            plt.title("Training and Validation Loss per Epoch", fontsize=5)
            plt.legend(fontsize=5)

            # Generate x-axis labels
            labels = training_monitor_notebook.generate_labels()

            plt.xticks(
                ticks=range(1, len(training_monitor_notebook.cumulative_epochs) + 1),
                labels=labels,
                fontsize=5
            )
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

        # Replace all negative values with zero
        y_pred = np.where(y_pred < 0, 0, y_pred)

        # Calculer le nRMSE
        nrmse_KPI = np.sqrt(mean_squared_error(y_valid, y_pred)) / (max_y_val - min_y_val)
        # st.write(f'NRMSE: {nrmse_KPI}')
        st.session_state.nrmse_value = nrmse_KPI

        # Calculer le RMSE
        rmse_KPI = np.sqrt(mean_squared_error(y_valid, y_pred))
        # st.write(f'RMSE: {rmse_KPI}')
        st.session_state.rmse_value = rmse_KPI

        # Calculer le MAE
        mae_KPI = round(mean_absolute_error(y_valid, y_pred), 5)
        # st.write(f'MAE: {mae_KPI}')
        st.session_state.mae_value = mae_KPI

        # Ensure st.session_state.model_info is initialized
        if "model_info" not in st.session_state:
            st.session_state.model_info = []

        # Update resultats.json structure
        conversion_factor = st.session_state.conversion_factor if "conversion_factor" in st.session_state else 1
        resultats = {
            "donnees_entrees": {
                "donnees_observees": {
                    "en_unite_mesure": y_valid.to_numpy().flatten().tolist(),  # Convert to NumPy array before flattening
                    "en_unite_mesure_2": (y_valid.to_numpy().flatten() * conversion_factor).tolist()  # Convert to NumPy array before flattening
                }
            },
            "resultats": {
                "predictions": {
                    "modeles": [
                        {
                            **modele,
                            "donnees_predites": {
                                "en_unite_mesure": y_pred.flatten().tolist(),
                                "en_unite_mesure_2": (y_pred.flatten() * conversion_factor).tolist()
                            }
                        }
                        for modele in st.session_state.model_info
                    ]
                }
            }
        }

        # Mettre à jour le modèle courant avec celui finetuné
        #Enregistrer les poids
        torch.save(model.state_dict(), dossier_modele_courant + "modele.pth")

        # Save scalers
        joblib.dump(x_scaler, dossier_modele_courant + "x_scaler.pkl")
        joblib.dump(y_scaler, dossier_modele_courant + "y_scaler.pkl")

        # Save model parameters (input_size, hidden_size, output_size, kpi)
        params = {
            "input_size": loaded_input_size,
            "hidden_size": loaded_hidden_size,
            "output_size": loaded_output_size,
            "kpi": {
                "mae": mae_KPI,
                "nrmse": nrmse_KPI,
                "rmse": rmse_KPI
            }
        }
        with open(dossier_modele_courant + "modele_parametres.json", "w") as f:
            json.dump(params, f)

    else:
        st.write("Aucun fichier validé disponible. Procédez au dépot et à la validation des données avant.")

    return 1