import os
import streamlit as st
import numpy as np
import pandas as pd
import shutil
import time
from menu import display_menu
from dependency_manager import check_dependencies
from utilitaires.preprocessing.Preprocessing_train import preprocesser_les_donnees
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
    Page qui s'affiche lorsque l'utilisateur choisit une prédiction avec création d'un modèle.
    Affiche l'interface de dépôt, validation et prétraitement des données.

    Cette fonction permet aux utilisateurs de déposer des fichiers de données, de valider leur format et de les prétraiter pour l'entraînement et la prédiction.
    Elle inclut également des boutons pour nettoyer les dossiers de validation et de résultats.

    Parameters:
    None

    Returns:
    None
    """
    afficher_bandeau_titre()
    st.title("Dépot, Validation et Prétraitement des Données")

    check_dependencies("Dépot et Validation des Données")

 
    st.markdown("#### Données pour l'entrainement du modèle")   # ajout Claire

    # Mise en forme et message de rappel sur le format attendu des données
    st.markdown(
    """
    <div style="background-color:#f5f5f5; padding:10px; border-radius:5px;border: 2px solid orange;">
        <em style="color:#333333; font-size:14px;">
        ⚠️ Attention :<br>
        Afin de procéder à l'entraînement du modèle, le jeu de données d'entraînement doit:
        <li>contenir obligatoirement <strong>deux colonnes</strong> comprenant la <strong>date</strong> et la <strong>valeur associée</strong></li>
        <li>la date doit être au format <strong>YYYY-MM-DD HH:MM:SS</strong></li>
        <li>la valeur de la deuxième colonne doit être <strong>numérique</strong></li>
        <li>le jeu de données ne doit pas contenir de <strong>valeurs manquantes (cellules vides)</strong></li>
        <br>Si les écarts de temps entre les différentes lignes sont trop importants, des erreurs lors du pré-traitement et de l'entraînement du modèle peuvent survenir.</strong></br>
        </em>
    </div>
    """,
    unsafe_allow_html=True
    )

    # Espace de dépôt des données d'entraînement
    uploaded_file = st.file_uploader("Déposez vos fichiers d'entraînement du modèle ici :", type=["csv", "txt"])

    # Lecture et affichage des données d'entraînement
    if uploaded_file is not None:
        try:
            input_data = pd.read_csv(uploaded_file)
            # Vérification des dimensions des data raw (Données du traffic réseau et du temps correspondant)
            if input_data.shape[0] < 1 or input_data.shape[1] != 2:
                st.error(f"Erreur : le fichier doit avoir 2 colonnes et au moins 1 ligne, mais a les dimensions suvantes : {input_data.shape}. Veuillez déposer un fichier valide.")
                input_data = None  # Annulation du fichier chargé
            else:
                st.write("Affichage d'un aperçu des données d'entrée pour l'entraînement :", input_data.head())
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier d'entraînement : {e}")
            input_data = None
    else:
        # Jeu de données par défaut afin de pouvoir effectuer des tests
        input_data = pd.read_csv(path_donnees_raw + "donnees_par_defaut/raw_train.csv")
        # st.write(f"Format des données d'entrée pour l'entrainement par défaut - Nombre de lignes: {input_data.shape[0]:,}, Nombre de colonnes: {input_data.shape[1]}")
        # st.write("Apperçu des données d'entrée pour l'entrainement par défaut:", input_data.head())

   # Date de la première observation dans la série des temps observés
    if "date_premiere_observation" not in st.session_state:
        st.session_state.date_premiere_observation = "2025-02-10 00:01:00"  # Default value

    st.markdown("#### Données pour la prédiction") # ajout Claire
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
    uploaded_file_2 = st.file_uploader("Déposez vos fichiers pour effectuer la prédiction ici :", type=["csv", "txt"])

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
    
    # Bouton pour valider les données
    if st.button("Valider"):
        with st.spinner("Préparation des données... (cette étape peut prendre plusieurs minutes en fonction de la taille du fichier déposé)", show_time=True):
            if input_data is not None and prediction_data is not None:
                if prediction_data.isna().values.any():
                    st.error("Erreur: Les données contiennent des valeurs manquantes.")
                elif (prediction_data < 0).values.any():
                    st.error("Erreur: Les données contiennent des valeurs négatives.")
                else:
                    # Preprocessing et Validation des données à utiliser pour l'entrainement
                    # Create the directory if it does not exist
                    os.makedirs(donnees_a_la_volee_dir, exist_ok=True)

                    preprocesser_les_donnees(donnees_a_la_volee_dir, input_data, horizon=horizon, sliding_window_train = sliding_window_train, sliding_window_valid = sliding_window_valid)
                    #preprocesser_les_donnees(donnees_a_la_volee_dir, input_data)

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
                    st.success("✅ Les données sont valides et pré-traitées. Vous pouvez passer à l'étape suivante.")

            else:
                st.error("Erreur: Aucun fichier n'a été téléchargé.")

if __name__ == "__main__":
    show()
