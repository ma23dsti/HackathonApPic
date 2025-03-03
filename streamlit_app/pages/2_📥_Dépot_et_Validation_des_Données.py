import streamlit as st
import numpy as np
import pandas as pd
from menu import display_menu
from dependency_manager import check_dependencies

display_menu()

def show():
    st.title("Dépot et Validation des Données")

    check_dependencies("Dépot et Validation des Données")

    st.write("Veuillez entrer les données pour les 60 dernières secondes:")

    # Espace de dépôt des données
    uploaded_file = st.file_uploader("Déposez vos fichiers ici", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            input_data = pd.read_csv(uploaded_file)
            st.write("Affichage des données d'entrée pour la prédiction :", input_data)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
            input_data = None
    else:
        # Dummy input data for testing
        input_data = np.random.rand(1, 60)
        st.write("Affichage des données d'entrée pour la prédiction :", input_data)

    if st.button("Validation des données"):
        if input_data is not None:
            # Validation des données
            if np.any(np.isnan(input_data)):
                st.error("Erreur: Les données contiennent des valeurs manquantes.")
            elif np.any(input_data < 0):
                st.error("Erreur: Les données contiennent des valeurs négatives.")
            else:
                st.success("Les données sont valides.")
                st.session_state.valid_depot_donnees = True
        else:
            st.error("Erreur: Aucun fichier n'a été téléchargé.")

if __name__ == "__main__":
    show()
