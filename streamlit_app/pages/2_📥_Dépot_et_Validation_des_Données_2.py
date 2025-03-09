import os
import streamlit as st
import numpy as np
import pandas as pd
import shutil
from menu import display_menu
from dependency_manager import check_dependencies
from utilitaires.Preprocessing_des_donnees import preprocesser_les_donnees_1, preprocesser_les_donnees_2

# Afficher le menu
display_menu()

# Dossiers de données
path_donnees_raw = "streamlit_app/static/donnees/donnees_raw/"    # Jeu de données par défaut (jeu de données fournis lors de la phase 1 de l'hackathon)
preprocessing_dir = "streamlit_app/static/donnees/donnees_preprocessees/"
donnees_a_la_volee_dir = os.path.join(preprocessing_dir, "donnees_a_la_volee/")
resultats_a_la_volee_dossier = "streamlit_app/resultats/donnees_a_la_volee/"

def show():
    
    st.title("Dépot, Validation et Prétraitement des Données")

    check_dependencies("Dépot et Validation des Données")

    if st.button("Nettoyer les dossiers de validation"):
        # Vérifier si le dossier existe
        if os.path.exists(donnees_a_la_volee_dir) and os.path.isdir(donnees_a_la_volee_dir):
            # Loop through all files in the directory and remove them
            for file_name in os.listdir(donnees_a_la_volee_dir):
                file_path = os.path.join(donnees_a_la_volee_dir, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # S'il y a un sous dossiers, le supprimer.
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
            print("Tous les fichiers de 'donnees_a_la_volee/' ont été supprimés.")
        else:
            print("Le dossier 'donnees_preprocessees/donnees_a_la_volee/' n'existe pas.")

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
        # Jeu de données par défaut afin de pouvoir effectuer des tests
        input_data = pd.read_csv(path_donnees_raw + "donnees_par_defaut/raw_train.csv")
        st.write(f"Format des données d'entrée par défaut - Nombre de lignes: {input_data.shape[0]:,}, Nombre de colonnes: {input_data.shape[1]}")
        st.write("Apperçu des données d'entrée par défaut:", input_data[:5])

    # Lecture et affichage des données de prédiction
    if uploaded_file_2 is not None:
        try:
            prediction_data = pd.read_csv(uploaded_file_2, header=None)
            st.write("Affichage des données d'entrée pour la prédiction :", prediction_data)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier de prédiction : {e}")
            prediction_data = None
    else:
        # Données fictives pour les tests
        prediction_data = np.random.rand(1, 60)
        st.write("Données de prédiction fictives:", prediction_data)

        # Jeu de données de test par défaut afin de pouvoir effectuer des tests
        prediction_data = pd.read_csv(preprocessing_dir + "donnees_par_defaut/x_valid_s65_o60_p5-1.csv", header=None)
        st.write(f"Format des données d'entrée par défaut pour la prédiction - Nombre de lignes: {prediction_data.shape[0]:,}, Nombre de colonnes: {input_data.shape[1]}")
        st.write("Apperçu des données d'entrée par défaut pour la prédiction:", prediction_data)

    # Bouton pour valider les données
    if st.button("Valider"):
        if input_data is not None and prediction_data is not None:
            if prediction_data.isna().values.any():
                st.error("Erreur: Les données contiennent des valeurs manquantes.")
            elif (prediction_data < 0).values.any():
                st.error("Erreur: Les données contiennent des valeurs négatives.")
            else:
                # Validation des données utilisées pour l'entrainement
                # Create the directory if it does not exist
                os.makedirs(donnees_a_la_volee_dir, exist_ok=True)
                donnees_raw_train, donnees_raw_valid = preprocesser_les_donnees_1(preprocessing_dir, input_data)
                window_size_x = 60
                window_size_y = 5
                step = 13
                subset = "train"
                preprocesser_les_donnees_2(donnees_a_la_volee_dir, donnees_raw_train, window_size_x=window_size_x, window_size_y=window_size_y, step=step, subset=subset)
                window_size_x = 60
                window_size_y = 5
                step = 65
                subset = "valid"
                preprocesser_les_donnees_2(donnees_a_la_volee_dir, donnees_raw_valid, window_size_x=window_size_x, window_size_y=window_size_y, step=step, subset=subset)

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
                st.success("Les données sont valides.")

        else:
            st.error("Erreur: Aucun fichier n'a été téléchargé.")

if __name__ == "__main__":
    show()
