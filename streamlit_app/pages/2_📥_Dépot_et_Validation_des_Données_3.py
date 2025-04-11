import os
import streamlit as st
import numpy as np
import pandas as pd
import json
from menu import display_menu
from dependency_manager import check_dependencies
from utilitaires.mise_page import afficher_bandeau_titre

# Afficher le menu
display_menu()

# Dossiers de données
path_donnees_raw = "streamlit_app/static/donnees/donnees_raw/"    # Jeu de données par défaut (jeu de données fournis lors de la phase 1 de l'hackathon)
preprocessing_dir = "streamlit_app/static/donnees/donnees_preprocessees/"
donnees_a_la_volee_dir = os.path.join(preprocessing_dir, "donnees_a_la_volee/")
resultats_a_la_volee_dossier = "streamlit_app/resultats/donnees_a_la_volee/"

# Récupérer les différentes tailles nécessaires pour les validations
horizon = st.session_state.horizon_predictions
taille_fenetre_observee = st.session_state.taille_fenetre_observee
sliding_window_train = st.session_state.sliding_window_train
sliding_window_valid = st.session_state.sliding_window_valid

def show():    

    """
    Page qui s'affiche lorque l'utilisateur choisit de faire une prédiction avec un modèle à upload
    Affiche l'interface de dépôt et validation des données.

    Cette fonction permet aux utilisateurs d'indiquer le dossier des fichiers d'un modèle et des fichiers de prédiction,
    de valider leur format et de les afficher. Elle inclut également des messages de validation pour guider
    l'utilisateur à travers les étapes de dépôt et de validation.

    Parameters:
    None

    Returns:
    None
    """
    afficher_bandeau_titre()
    st.title("Dépot et Validation des Données")

    check_dependencies("Dépot et Validation des Données")

    st.markdown("#### Modèle pour la prédiction")   # ajout Claire

        # Field to select a folder (mandatory)
    if "model_charge" not in st.session_state:
        st.session_state.model_charge = None

    st.write("Entrez le chemin du dossier contenant le modèle :")

    fichiers_modele = ["modele.pth", "modele_parametres.json", "x_scaler.pkl", "y_scaler.pkl"]

    # Text input for folder path
    folder_path_input = st.text_input(
        "Chemin du dossier :",
        value=st.session_state.get('model_charge', ""),  # Use the stored value if available
        help="Entrez le chemin complet du dossier contenant le modèle."
    )

    # Update the session state only if the input changes
    if folder_path_input and folder_path_input != st.session_state.get('model_charge', ""):
        if os.path.isdir(folder_path_input):
            missing_files = [f for f in fichiers_modele if not os.path.isfile(os.path.join(folder_path_input, f))]
            if missing_files:
                st.error(f"Erreur : Le dossier spécifié est invalide. Les fichiers suivants sont manquants : {', '.join(missing_files)}")
                return

            # Load and validate model parameters
            param_file_path = os.path.join(folder_path_input, "modele_parametres.json")
            try:
                with open(param_file_path, "r") as f:
                    params = json.load(f)
                loaded_input_size = params["input_size"]
                loaded_hidden_size = params["hidden_size"]
                loaded_output_size = params["output_size"]

                if loaded_input_size != taille_fenetre_observee:
                    st.error(f"Erreur : input_size attendu ({taille_fenetre_observee}) ne correspond pas à celui du modèle ({loaded_input_size}). Fichier : {param_file_path}")
                    return
                if loaded_hidden_size != 800:
                    st.error(f"Erreur : hidden_size attendu (800) ne correspond pas à celui du modèle ({loaded_hidden_size}). Fichier : {param_file_path}")
                    return
                if loaded_output_size != horizon:
                    st.error(f"Erreur : output_size attendu ({horizon}) ne correspond pas à celui du modèle ({loaded_output_size}). Fichier : {param_file_path}")
                    return
            except Exception as e:
                st.error(f"Erreur lors de la lecture ou de la validation de {param_file_path} : {e}")
                return

            st.session_state['model_charge'] = folder_path_input
        else:
            st.error("Erreur : Le chemin du dossier spécifié n'existe pas ou n'est pas un dossier.")
            return

    # Set default model path if no folder is entered and no value is stored
    if not st.session_state.get('model_charge'):
        default_model_path = f"streamlit_app/static/modeles/modele_par_defaut/modele_par_defaut_o{taille_fenetre_observee}_p{horizon}/"
        if os.path.isdir(default_model_path):
            missing_files = [f for f in fichiers_modele if not os.path.isfile(os.path.join(default_model_path, f))]
            if missing_files:
                st.error(f"Erreur : Le modèle par défaut est invalide. Les fichiers suivants sont manquants : {', '.join(missing_files)}")
                return
            st.session_state['model_charge'] = default_model_path
            st.info(f"Aucun modèle spécifié. Le modèle par défaut sera utilisé : {default_model_path}")
        else:
            st.error(f"Erreur : Le modèle par défaut n'existe pas au chemin spécifié : {default_model_path}")
            return

    if st.session_state.get('model_charge'):
        st.write(f"Modèle chargé depuis : {st.session_state.model_charge}")


    st.markdown("#### Données pour la prédiction") # ajout Claire    

    # Date de la première observation dans la série des temps observés
    if "date_premiere_observation" not in st.session_state:
        st.session_state.date_premiere_observation = "2025-02-10 00:01:00"  # Default value

    date_premiere_observation = st.text_input(
        "Date de la première observation (format: YYYY-MM-DD HH:MM:SS)",
        value=st.session_state.date_premiere_observation,
        help="Entrez une date au format 'YYYY-MM-DD HH:MM:SS'. Exemple: '2025-02-10 00:01:00'."
    )

    # Validate the date format and update session state
    try:
        pd.to_datetime(date_premiere_observation, format="%Y-%m-%d %H:%M:%S")
        st.session_state.date_premiere_observation = date_premiere_observation
    except ValueError:
        st.error("Erreur: La date doit être au format 'YYYY-MM-DD HH:MM:SS'. Exemple: '2025-02-10 00:01:00'.")
        return

    #st.write("Veuillez entrer les données pour les ", taille_fenetre_observee, " dernières secondes:")
    st.write(f"Veuillez entrer les données pour les {taille_fenetre_observee} dernières secondes:")

    # Mise en forme et message de rappel sur le format attendu des données
    st.markdown(
    f"""
    <div style="background-color:#f5f5f5; padding:10px; border-radius:5px;border: 2px solid orange;">
        <em style="color:#333333; font-size:14px;">
        ⚠️ Attention :<br>
        Le format attendu pour ce fichier est <strong>une ligne unique</strong> contennant <strong>{taille_fenetre_observee} observations</strong> afin de faire la prédiction.<br>
        Si vous importez un fichier ne respectant pas ce format, des erreurs peuvent survenir.<br>
        </em>
    </div>
    """,
    unsafe_allow_html=True
    )

    # Espace de dépôt des données de prédiction
    uploaded_file_2 = st.file_uploader("Déposez vos fichiers de prédiction ici :", type=["csv", "txt"])

    # Lecture et affichage des données de prédiction
    if uploaded_file_2 is not None:
        try:
            prediction_data = pd.read_csv(uploaded_file_2, header=None)
            # Vérification des dimensions des données observées afin de pouvoir effectuer la prédiction
            expected_shape_2 = (1, taille_fenetre_observee)
            if prediction_data.shape != expected_shape_2:
                st.error(f"Erreur : le fichier doit avoir les dimension suivantes : {expected_shape_2}, mais a {prediction_data.shape}. Veuillez déposer un fichier valide.")
                prediction_data = None  # Annulation du fichier chargé
            else:
                st.write("Affichage des données d'entrée pour la prédiction :", prediction_data)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier de prédiction : {e}")
            prediction_data = None
    else:
        # Jeu de données de test par défaut afin de pouvoir effectuer des tests
        prediction_data = pd.read_csv(preprocessing_dir + "donnees_par_defaut/x_valid_s" + str(sliding_window_valid) + "_o" + str(taille_fenetre_observee) + "_p" + str(st.session_state.horizon_predictions) + "-1.csv", header=None)
        #st.write(f"Format des données d'entrée par défaut pour la prédiction - Nombre de lignes: {prediction_data.shape[0]:,}, Nombre de colonnes: {prediction_data.shape[1]}")
        #st.write("Apperçu des données d'entrée par défaut pour la prédiction:", prediction_data)

    # Bouton pour valider les données
    if st.button("Valider"):

            if prediction_data.isna().values.any():
                st.error("Erreur: Les données contiennent des valeurs manquantes.")
            elif (prediction_data < 0).values.any():
                st.error("Erreur: Les données contiennent des valeurs négatives.")
            else:
                # Preprocessing et Validation des données à utiliser pour l'entrainement
                # Create the directory if it does not exist
                os.makedirs(donnees_a_la_volee_dir, exist_ok=True)



                # Validation des données utilisées pour la prédiction
                # Convert prediction_data to a DataFrame with a 'value' column
                prediction_data_df = pd.DataFrame({'value': prediction_data.values.flatten()})
                prediction_data_df.index = range(1, len(prediction_data_df) + 1)

                # Store properly formatted data in session state
                st.session_state.prediction_data = prediction_data_df

                st.session_state.prediction_effectuee = False
                dernieres_donnees_deposees = prediction_data_df
                if ('precedentes_donnees_deposees' not in st.session_state) or ('precedentes_donnees_deposees' in st.session_state and not dernieres_donnees_deposees.equals(st.session_state.precedentes_donnees_deposees)):
                    st.session_state.nouveau_depot_donnees = True
                st.session_state.precedentes_donnees_deposees = dernieres_donnees_deposees              
                st.session_state.valid_depot_donnees = True

                st.session_state.prediction_avec_model_charge = True

                st.success("Les données sont valides.")


if __name__ == "__main__":
    show()
