import streamlit as st
import numpy as np
import pandas as pd
from menu import display_menu
from dependency_manager import check_dependencies

# Afficher le menu
display_menu()

def show():
    st.title("Dépot et Validation des Données")

    check_dependencies("Dépot et Validation des Données")

    st.write("Veuillez entrer les données pour les 60 dernières secondes:")

    # Espace de dépôt des données d'entraînement
    uploaded_file = st.file_uploader("Déposez vos fichiers d'entraînement du modèle ici :", type=["csv", "txt"])

    # Espace de dépôt des données de prédiction
    uploaded_file_2 = st.file_uploader("Déposez vos fichiers de prédiction ici :", type=["csv", "txt"])

    # Lecture et affichage des données d'entraînement
    if uploaded_file is not None:
        try:
            input_data = pd.read_csv(uploaded_file)
            st.write("Affichage d'un aperçu des données d'entrée pour l'entraînement :", input_data.head()) # vérifier le comportement avec un dataset volumineux
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier d'entraînement : {e}")
            input_data = None
    else:
        # Données fictives pour les tests
        input_data = np.random.rand(1, 60)
        st.write("Données d'entraînement fictives:", input_data)

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
        if input_data is not None and prediction_data is not None:
            st.success("Les données ont été validées avec succès.")
            st.session_state.valid_depot_donnees = True
        else:
            st.error("Erreur: Aucun fichier n'a été téléchargé.")

if __name__ == "__main__":
    show()
