import json
import os
import shutil
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from menu import display_menu
from dependency_manager import check_dependencies
from utilitaires.Prediction import predire_le_traffic
from utilitaires.Resultat import mettre_a_jour_model_info
from utilitaires.mise_page import afficher_bandeau_titre
from streamlit_extras.add_vertical_space import add_vertical_space

display_menu()

preprocessing_dir = "streamlit_app/static/donnees/donnees_preprocessees/"
dossier_donnees_pour_entrainement = preprocessing_dir + "donnees_a_la_volee/"

def show():
    """
    Affiche l'interface de prédiction.

    Cette fonction permet aux utilisateurs de faire des prédictions de trafic réseau en utilisant un modèle pré-entraîné.
    Elle inclut les étapes de validation des données d'entrée, de génération des prédictions, et d'affichage des résultats
    sous forme de tableau et de graphiques. Elle fournit également des options pour télécharger les prédictions en CSV.

    Parameters:
    None

    Returns:
    None
    """
    afficher_bandeau_titre()
    st.title("Prédictions")
    add_vertical_space(2)
    check_dependencies("Prédictions") 

    # Define dossier_modele_courant at the start of the function
    dossier_modele_courant = "streamlit_app/static/modeles/modele_courant/"

    # Tailles nécessaires pour la prédiction'
    horizon = st.session_state.horizon_predictions
    taille_fenetre_observee = st.session_state.taille_fenetre_observee

    # Si un import de fichier a été fait pour l'historique des données dont on veut effectuer la prédiction,
    # alors l'historique des précédentes prédictions (effectuées sur d'autres données) est effacé.
    if 'nouveau_depot_donnees' in st.session_state:
        if st.session_state.nouveau_depot_donnees and 'resultats' in st.session_state:
            st.session_state.resultats["resultats"]["predictions"].pop("modeles", None)
            # Besoin de supprimer les resultats dans la sous structure model_info egalement pour ne pas garder un lien avec les anciens resultats.
            st.session_state.model_info = []

    # S'assurer que les données sont sous forme de DataFrame avec une colonne 'value'
    if not isinstance(st.session_state.prediction_data, pd.DataFrame) or 'value' not in st.session_state.prediction_data.columns:
        st.session_state.prediction_data = pd.DataFrame({'value': st.session_state.prediction_data.values.flatten()})
        st.session_state.prediction_data.index = range(1, len(st.session_state.prediction_data) + 1)

    # Bouton pour faire les prédictions
    if st.button("Faire une prédiction avec le modèle"):

        if 'predictions_df' not in st.session_state or not st.session_state.prediction_effectuee:
            # Flatten and reshape to (1, 60) for model prediction
            prediction_data_reshaped = np.array(st.session_state.prediction_data).flatten().reshape(1, -1)
            # Ensure we have 60 features
            if prediction_data_reshaped.shape[1] != taille_fenetre_observee:
                st.error(f"Erreur: Le modèle attend {taille_fenetre_observee} colonnes, mais {prediction_data_reshaped.shape[1]} ont été détectées.")
                return

            if st.session_state.prediction_avec_model_charge:
                # Ensure model_charge is not None
                if not st.session_state.model_charge:
                    # Set default model path
                    default_model_path = "streamlit_app/static/modeles/modele_par_defaut/modele_par_defaut_restreint_o60_p5/"
                    if os.path.isdir(default_model_path):
                        st.session_state.model_charge = default_model_path
                        st.info(f"Aucun modèle chargé. Le modèle par défaut sera utilisé : {default_model_path}")
                    else:
                        st.error(f"Erreur : Le modèle par défaut n'existe pas au chemin spécifié : {default_model_path}")
                        return

                # Load the model from the provided path directory to the current model directory
                fichiers_modele = ["modele.pth", "modele_parametres.json", "x_scaler.pkl", "y_scaler.pkl"]
                os.makedirs(dossier_modele_courant, exist_ok=True)

                # Copier chaque fichier dans le dossier du modèle courant
                for fichier in fichiers_modele:
                    chemin_source = os.path.join(st.session_state.model_charge, fichier)
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

            predictions = predire_le_traffic(prediction_data_reshaped)
            st.success(" Prédiction effectuée avec succès.", icon="✅")

            predictions = np.array(predictions).flatten()
            # Check lengths to prevent errors
            if len(predictions) != horizon:  # Expected next horizon values in time series
                st.error(f"Erreur: Le modèle a généré {len(predictions)} valeurs, mais {horizon} étaient attendues.")
                return

            # Create predictions DataFrame
            start_index = prediction_data_reshaped.shape[1] + 1
            predictions_df = pd.DataFrame({
                'Index': np.arange(start_index, start_index + len(predictions)),
                'Predictions': predictions
            })

            st.session_state.predictions_df = predictions_df
            st.session_state.prediction_effectuee = True
            st.session_state.valid_predictions = True



    # Afficher les prédictions et le graphique même si le bouton n'est pas recliqué
    if 'predictions_df' in st.session_state and st.session_state.prediction_effectuee:

        if ('prediction_historique_recalculee' not in st.session_state) or (st.session_state.prediction_historique_recalculee == False) or (st.session_state.nouveau_depot_donnees==True):
            resultats = mettre_a_jour_model_info(st.session_state.predictions_df['Predictions'])
            st.session_state.prediction_historique_recalculee = True
            st.session_state.nouveau_depot_donnees = False

        # Récupération des données depuis st.session_state
        donnees_entrees = st.session_state.resultats["donnees_entrees"]
        donnees_predictions = st.session_state.resultats["resultats"]["predictions"]
        # Extraction des données observées
        x_observees = donnees_entrees["temps_relatif"]
        y_observees = donnees_entrees["donnees_observees"]["en_unite_mesure"]  # Access "en_unite_mesure"

        # Ensure x_observees and y_observees are lists or NumPy arrays
        x_observees = np.array(x_observees) if isinstance(x_observees, list) else x_observees
        y_observees = np.array(y_observees) if isinstance(y_observees, list) else y_observees

        # Transform unhashable types in y_observees
        y_observees = [str(val) if isinstance(val, dict) else val for val in y_observees]
        # Convert y_observees to NumPy array
        y_observees = np.array(y_observees)

        # Extraction des données prédites
        x_predictions = donnees_predictions["temps_relatif"]
        modeles = donnees_predictions["modeles"]

        # Afficher la prédiction du dernier modèle entrainé.
        plt.figure(figsize=(12, 6))
        plt.plot(
            x_observees, 
            y_observees, 
            label="Données observées", 
            color="blue", 
            linestyle="-"
        )
        #plt.plot(st.session_state.prediction_data.index, st.session_state.prediction_data['value'], label="Données d'entrée", color='blue')
        if 'entrainement_modele' in st.session_state:
            if st.session_state.entrainement_modele==True:
                nrmse_courant = st.session_state.nrmse_value
        else: 
            with open(os.path.join(dossier_modele_courant, "modele_parametres.json"), "r") as f:
                params = json.load(f)
                nrmse_courant = params["kpi"]["nrmse"]
        plt.plot(
            st.session_state.predictions_df['Index'],
            st.session_state.predictions_df['Predictions'],
            label=f'Prédictions - NRMSE {nrmse_courant:.4f}',
            color='red',
            linestyle="--",
            marker='o',  # Use circles as markers
            markersize=2  # Adjust marker size
        )
        # Add a line connecting the last observed point to the first predicted point
        plt.plot(
            [x_observees[-1], st.session_state.predictions_df['Index'].iloc[0]],
            [y_observees[-1], st.session_state.predictions_df['Predictions'].iloc[0]],
            color="red",
            linestyle="--"
        )
        plt.axvline(x=len(st.session_state.prediction_data), color='black', linestyle='--')
        plt.xlabel("Index")
        plt.ylabel("Valeur")
        plt.title("Prédiction du dernier modèle entrainé")
        plt.legend()
        st.pyplot(plt)

        # Afficher toutes les prédictions des différents modèles obtenus après entrainements.
        plt.figure(figsize=(12, 6))
        plt.plot(x_observees, y_observees, label="Données observées", color="blue", linestyle="-")
        # Fonction pour générer une couleur aléatoire en excluant le rouge
        def generate_random_color():
            while True:
                # Générer une couleur aléatoire (les composantes RVB sont entre 0 et 255)
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)

                # Vérifier si la couleur est trop rouge (par exemple, r > 200 pour éviter une couleur trop rouge)
                if r > 200 and g < 100 and b < 100:
                    continue  # Si c'est trop rouge, recommence la génération

                # Convertir la couleur en hexadécimal
                color = "#{:02x}{:02x}{:02x}".format(r, g, b)
                
                # Retourner la couleur générée si elle n'est pas rouge
                return color
        # Générer une liste de couleurs aléatoires pour les prédictions des différents modèles entrainés
        couleurs = [generate_random_color() for _ in range(len(modeles) - 1)]
        # Ajouter le rouge comme dernière couleur pour le modèle moyen
        couleurs.append("#ff0000")  # La dernière couleur est le rouge
        for i, modele in enumerate(modeles):
            plt.plot(
                x_predictions,
                modele["donnees_predites"]["en_unite_mesure"],
                label=f'Prédictions - {modele["nom"]} - NRMSE {modele["kpi"]["nrmse"]:.4f}',
                color=couleurs[i % len(couleurs)],
                linestyle="--",
                marker='x',  # Use crosses as markers
                markersize=2  # Adjust marker size
            )
        # Ajouter une ligne verticale pour séparer observations et prédictions
        plt.axvline(x=max(x_observees), color="black", linestyle="--")
        plt.xlabel("Temps relatif")
        plt.ylabel("Valeur")
        plt.title("Prédictions des différents modèles obtenus après entrainements")
        plt.legend()
        st.pyplot(plt)

        st.write("### Prédictions générées:")
        st.write(st.session_state.predictions_df)

    # Ajouter une séparation
    st.markdown("---")

    if st.session_state.prediction_effectuee:
        # Téléchargement des prédictions en CSV
        csv = st.session_state.predictions_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger les prédictions en CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    show()