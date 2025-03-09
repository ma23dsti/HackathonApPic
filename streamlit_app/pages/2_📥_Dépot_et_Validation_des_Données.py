import os
import streamlit as st
import pandas as pd
import shutil
from menu import display_menu
from dependency_manager import check_dependencies

display_menu()

# Dossiers de données
preprocessing_dir = "streamlit_app/static/donnees/donnees_preprocessees/"

def show():
    st.title("Dépot et Validation des Données")

    ###if 'prediction_data'  in st.session_state:
        ###st.write(st.session_state.prediction_data)

    check_dependencies("Dépot et Validation des Données")

    resultats_a_la_volee_dossier = "streamlit_app/resultats/donnees_a_la_volee/"
    if st.button("Nettoyer le dossier des resultats"):
        # Vérifier si le dossier existe
        if os.path.exists(resultats_a_la_volee_dossier) and os.path.isdir(resultats_a_la_volee_dossier):
            # Loop through all files in the directory and remove them
            for file_name in os.listdir(resultats_a_la_volee_dossier):
                file_path = os.path.join(resultats_a_la_volee_dossier, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # S'il y a un sous dossiers, le supprimer.
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
            print("Tous les fichiers de 'donnees_a_la_volee/' ont été supprimés.")
        else:
            print("Le dossier 'resultats/donnees_a_la_volee/' n'existe pas.")

    st.write("Veuillez entrer les données pour les 60 dernières secondes:")

    # Espace de dépôt des données
    uploaded_file = st.file_uploader("Déposez vos fichiers ici", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            prediction_data = pd.read_csv(uploaded_file, header=None)
            st.write("Affichage des données d'entrée pour la prédiction :", prediction_data)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
            prediction_data = None
    else:
        # Jeu de données de test par défaut afin de pouvoir effectuer des tests
        prediction_data = pd.read_csv(preprocessing_dir + "donnees_par_defaut/x_valid_s65_o60_p5-1.csv", header=None)
        st.write(f"Format des données d'entrée par défaut pour la prédiction - Nombre de lignes: {prediction_data.shape[0]:,}, Nombre de colonnes: {prediction_data.shape[1]}")
        st.write("Apperçu des données d'entrée par défaut pour la prédiction:", prediction_data)

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
                st.success("Les données sont valides.")

        else:
            st.error("Erreur: Aucun fichier n'a été téléchargé.")

if __name__ == "__main__":
    show()
