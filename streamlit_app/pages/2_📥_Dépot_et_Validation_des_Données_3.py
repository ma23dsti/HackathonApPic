import streamlit as st
import numpy as np
import pandas as pd
import torch
from menu import display_menu
from dependency_manager import check_dependencies

# Afficher le menu
display_menu()

def show():
    st.title("Dépot et Validation des Données")

    check_dependencies("Dépot et Validation des Données")


    st.write("Veuillez entrer les données pour les 60 dernières secondes:")

    # Espace de dépôt des données d'entraînement
    uploaded_file = st.file_uploader("Déposez votre modèle au format PTH ici :", type=["pth"])

    # Espace de dépôt des données de prédiction
    uploaded_file_2 = st.file_uploader("Déposez vos fichiers de prédiction ici :", type=["csv", "txt"])

    # Charger et afficher le modèle
    if uploaded_file is not None:
        try:
            model = torch.load(uploaded_file)
            st.write("Modèle chargé avec succès.")
        except Exception as e:
            st.error(f"Erreur lors de l'importation du modèle : {e}")
            model = None
    else:
        # Modèle fictif pour les tests
        st.write("Aucun modèle chargé.")
        model = True

    # Lecture et affichage des données de prédiction
    if uploaded_file_2 is not None:
        try:
            prediction_data = pd.read_csv(uploaded_file_2)
            st.write("Affichage des données d'entrée pour la prédiction :", prediction_data)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier de prédiction : {e}")
            prediction_data = None
    else:
        # Données fictives pour les tests
        prediction_data = np.random.rand(1, 60)
        st.write("Données de prédiction fictives:", prediction_data)

    # Bouton pour valider les données
    if st.button("Valider"):
        if model is not None and prediction_data is not None:
            st.success("Les données ont été validées avec succès.")
            st.session_state.valid_depot_donnees = True
        else:
            st.error("Erreur: Aucun fichier n'a été téléchargé ou les fichiers sont invalides.")

if __name__ == "__main__":
    show()
