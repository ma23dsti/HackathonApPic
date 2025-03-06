import streamlit as st
import numpy as np
import pandas as pd
from menu import display_menu
from dependency_manager import check_dependencies

display_menu()

# Dossiers de données
preprocessing_dir = "streamlit_app/static/dossier_donnees/donnees_preprocessees/"

def show():
    st.title("Dépot et Validation des Données")

    ###if 'input_data'  in st.session_state:
        ###st.write(st.session_state.input_data)

    check_dependencies("Dépot et Validation des Données")

    st.write("Veuillez entrer les données pour les 60 dernières secondes:")

    # Espace de dépôt des données
    uploaded_file = st.file_uploader("Déposez vos fichiers ici", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            input_data = pd.read_csv(uploaded_file, header=None)
            st.write("Affichage des données d'entrée pour la prédiction :", input_data)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
            input_data = None
    else:
        # Jeu de données de test par défaut afin de pouvoir effectuer des tests
        input_data = pd.read_csv(preprocessing_dir + "donnees_par_defaut/x_valid_s65_o60_p5-1.csv", header=None)
        st.write(f"Format des données d'entrée par défaut pour la prédiction - Nombre de lignes: {input_data.shape[0]:,}, Nombre de colonnes: {input_data.shape[1]}")
        st.write("Apperçu des données d'entrée par défaut pour la prédiction:", input_data)

    if st.button("Validation des données"):
        if input_data is not None:
            if input_data.isna().values.any():
                st.error("Erreur: Les données contiennent des valeurs manquantes.")
            elif (input_data < 0).values.any():
                st.error("Erreur: Les données contiennent des valeurs négatives.")
            else:
                # Convert input_data to a DataFrame with a 'value' column
                input_data_df = pd.DataFrame({'value': input_data.values.flatten()})
                input_data_df.index = range(1, len(input_data_df) + 1)

                # Store properly formatted data in session state
                st.session_state.input_data = input_data_df

                st.success("Les données sont valides.")
                st.session_state.prediction_effectuee = False
                st.session_state.valid_depot_donnees = True
        else:
            st.error("Erreur: Aucun fichier n'a été téléchargé.")




if __name__ == "__main__":
    show()
