import os
import shutil
import streamlit as st
import pandas as pd
import numpy as np
from menu import display_menu
from dependency_manager import check_dependencies
from utilitaires.mise_page import afficher_bandeau_titre

display_menu()

# Dossiers de données
preprocessing_dir = "streamlit_app/static/donnees/donnees_preprocessees/"

# Récupérer les différentes tailles nécessaires pour les validations
horizon = st.session_state.horizon_predictions
taille_fenetre_observee = st.session_state.taille_fenetre_observee
sliding_window_valid = st.session_state.sliding_window_valid

def show():
    """
    Page qui s'affiche quand l'utilisateur choisit de prédire avec un modele baseline.
    Affiche l'interface de dépôt et validation des données.

    Cette fonction permet aux utilisateurs de déposer des fichiers de données pour la prédiction,
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

    st.write(f"Veuillez entrer les données pour les {taille_fenetre_observee} dernières secondes.")

    # Espace de dépôt des données
    uploaded_file = st.file_uploader("Déposez vos fichiers ici :", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            prediction_data = pd.read_csv(uploaded_file, header=None)
            # Vérification des dimensions des données observées
            expected_shape = (1, taille_fenetre_observee)
            if prediction_data.shape != expected_shape:
                st.error(f"Erreur : le fichier doit avoir les dimensions suivantes : {expected_shape}, mais a {prediction_data.shape}. Veuillez déposer un fichier valide.")
                prediction_data = None
            else:
                st.write("Affichage des données d'entrée pour la prédiction :", prediction_data)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
            prediction_data = None
    else:
        # Jeu de données de test par défaut afin de pouvoir effectuer des tests
        prediction_data = pd.read_csv(preprocessing_dir + f"donnees_par_defaut/x_valid_s{sliding_window_valid}_o{taille_fenetre_observee}_p{horizon}-1.csv", header=None)
        st.write(f"Apperçu des données d'entrée par défaut pour la prédiction (taille fenêtre: {taille_fenetre_observee}, horizon: {horizon}):", prediction_data.head())

    # Bouton pour valider les données
    if st.button("Validation des données"):
        if prediction_data is not None:
            if prediction_data.isna().values.any():
                st.error("Erreur: Les données contiennent des valeurs manquantes.")
            elif (prediction_data < 0).values.any():
                st.error("Erreur: Les données contiennent des valeurs négatives.")
            else:
                # Convert prediction_data to a DataFrame with a 'value' column
                prediction_data_df = pd.DataFrame({'value': prediction_data.values.flatten()})
                prediction_data_df.index = range(1, len(prediction_data_df) + 1)

                # Store properly formatted data in session state
                st.session_state.prediction_data = prediction_data_df
                st.session_state.prediction_effectuee = False
                st.session_state.nouveau_depot_donnees = True
                st.session_state.valid_depot_donnees = True

                if st.session_state.premiere_prediction_seule:
                    # Dossier du modèle par défaut
                    dossier_modele_par_defaut = f"streamlit_app/static/modeles/modele_par_defaut/modele_par_defaut_restreint_o{taille_fenetre_observee}_p{horizon}/"
                    fichiers_modele = ["modele.pth", "modele_parametres.json", "x_scaler.pkl", "y_scaler.pkl"]
                    dossier_modele_courant = "streamlit_app/static/modeles/modele_courant/"
                    # Créer le dossier du modèle courant s'il n'existe pas.
                    os.makedirs(dossier_modele_courant, exist_ok=True)

                    # Copier les fichiers du modèle par défaut
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

                    st.session_state['premiere_prediction_seule'] = False

                st.success("✅ Les données sont valides.")
        else:
            st.error("Erreur: Aucun fichier n'a été téléchargé.")

if __name__ == "__main__":
    show()
