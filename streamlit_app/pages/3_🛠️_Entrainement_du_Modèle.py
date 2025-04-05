import shutil
import streamlit as st
import torch
import torch.nn as nn
from menu import display_menu
import time
from datetime import datetime
import os
from dependency_manager import check_dependencies

# Importer les fonctions utiles
from utilitaires.Entrainement import entrainer_le_modèle

# Afficher le menu
display_menu()

# Récupérer les différentes tailles pour le nommage des modèles
horizon = st.session_state.horizon_predictions
taille_fenetre_observee = st.session_state.taille_fenetre_observee

def show():
    st.title("Entraînement du Modèle")

    check_dependencies("Entraînement du Modèle")

    preprocessing_dir = "streamlit_app/static/donnees/donnees_preprocessees/"
    dossier_donnees_pour_entrainement = preprocessing_dir + "donnees_a_la_volee/"

    st.write("Cliquez sur le bouton ci-dessous pour lancer l'entraînement du modèle.")

    # Initialiser les variables d'état
    if 'total_steps' not in st.session_state:
        st.session_state.total_steps = 0
    if 'nrmse_value' not in st.session_state:
        st.session_state.nrmse_value = 0.0
    if 'dummy_model_path' not in st.session_state:
        st.session_state.dummy_model_path = ""
    if 'valid_entrainement' not in st.session_state:
        st.session_state.valid_entrainement = False

    # Valeur de baseline pour un modèle de qualité moyenne/décente
    baseline_nrmse = 0.15

    dossier_modeles_entraines = "streamlit_app/static/modeles/modeles_entraines/"  # Ensure this is defined here

    # Bouton pour lancer l'entraînement
    if st.button("Lancer l'entraînement"):

        # Si un import de fichier a été fait pour l'historique des données dont on veut effectuer la prédiction,
        # alors l'historique des précédentes prédictions (effectuées sur d'autres données) est effacé.
        if 'nouveau_depot_donnees' in st.session_state:
            if st.session_state.nouveau_depot_donnees and 'resultats' in st.session_state:
                st.session_state.resultats["resultats"]["predictions"].pop("modeles", None)
                # Besoin de supprimer les resultats dans la sous structure model_info egalement pour ne pas garder un lien avec les anciens resultats.
                st.session_state.model_info = []
                # st.json(st.session_state.resultats)
                # nouveau_depot_donnees mis à Faux pour ne pas cleaner les données de résultats lors de le prochaine étape du flow: les prédictions.
                st.session_state.nouveau_depot_donnees = False

        nb_pred_max = 10
        nb_pred_max_atteint = False
        if 'resultats' in st.session_state:
            if "modeles" in st.session_state.resultats["resultats"]["predictions"]:
                if len(st.session_state.resultats["resultats"]["predictions"]["modeles"]) > nb_pred_max:
                    nb_pred_max_atteint = True
                    st.write("Nombre maximum de prédictions avec différents modèles atteint:", nb_pred_max, ". Pour effectuer d'autres prévisions, veuillez déposer un autre jeu de données observés ou revenez sur la page d'acceuil.")

        if not nb_pred_max_atteint or not ("modeles" in st.session_state.resultats["resultats"]["predictions"]) or (st.session_state.resultats["resultats"]["predictions"]["modeles"]==[]):
            # Créer le dossier pour les modèles entrainés s'il n'existe pas.
            os.makedirs(dossier_donnees_pour_entrainement, exist_ok=True)

            entrainer_le_modèle(dossier_donnees_pour_entrainement)

            # Sauvegarder les fichiers du modèle entrainé
            fichiers_modele = ["modele.pth", "modele_parametres.json", "x_scaler.pkl", "y_scaler.pkl"]
            dossier_modele_courant = "streamlit_app/static/modeles/modele_courant/"
            # Créer le dossier de sauvegarde du modèle courant s'il n'existe pas.
            dossier_modele_entraine = f"{dossier_modeles_entraines}modele_entraine_o{taille_fenetre_observee}_p{horizon}_{st.session_state.nrmse_value:.4f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
            os.makedirs(dossier_modele_entraine, exist_ok=True)
            # Copier les fichiers du modèle entrainé
            for fichier in fichiers_modele:
                chemin_source = os.path.join(dossier_modele_courant, fichier)
                chemin_destination = os.path.join(dossier_modele_entraine, fichier)

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

            ##progress_bar.empty()  # Supprimer la progress bar après la fin de l'entraînement
            st.success("Entraînement terminé.")

            # Après un (ré)entrainement de modèle, l'historique des resultats aura besoin d'être recalculé lors de la prochaine prédiction.
            if 'prediction_historique_recalculee' in st.session_state:
                st.session_state.prediction_historique_recalculee = False

            # Marquer l'entraînement comme terminé
            st.session_state.valid_entrainement = True

    # Séparation pour la section des métriques
    if st.session_state.valid_entrainement:
        st.divider()

        # Section "Métrique"
        st.header("Evaluation de la qualité du modèle")
        st.write(f"**NRMSE (Normalized Root Mean Square Error) :** {st.session_state.nrmse_value:.4f}")
        with st.expander("Qu'est ce que le **NRMSE** ?"):
            st.write("""Le **NRMSE** est une mesure de l'erreur normalisée entre les prédictions et les valeurs réelles. C'est une métrique couramment utilisée pour l'évaluation de ce type de modèle.
                 Plus d'informations en cliquant [ici](https://docs.oracle.com/cloud/help/fr/pbcs_common/PFUSU/insights_metrics_RMSE.htm#PFUSU-GUID-FD9381A1-81E1-4F6D-8EC4-82A6CE2A6E74).""")

        st.write(st.session_state.nrmse_value)
        # Estimation de la qualité du modèle par rapport à la base line
        if st.session_state.nrmse_value < baseline_nrmse * 0.9:
            st.success("La qualité du modèle est **bonne**. ✅")
        elif baseline_nrmse * 0.9 <= st.session_state.nrmse_value <= baseline_nrmse * 1.1:
            st.warning("La qualité du modèle est **moyenne**. ⚠️")
        else:
            st.error("La qualité du modèle est **mauvaise**. ❌")

        # Tooltip d'aide
        with st.expander("Comment la qualité du modèle est-elle déterminée ?"):
            st.write("""
                La qualité du modèle est évaluée en comparant le NRMSE (Normalized Root Mean Square Error) du modèle entraîné
                avec le NRMSE du modèle de référence pré-chargé sur la plateforme (valeur de référence: """, baseline_nrmse, """).
                - Si le NRMSE du modèle est inférieur à 90% du NRMSE du modèle de référence, la qualité est considérée comme **bonne** ✅.
                - Si le NRMSE est dans la plage [90% du NRMSE du modèle de référence, 110% du NRMSE du modèle de référence], la qualité est considérée comme **moyenne** ⚠️.
                - Si le NRMSE est supérieur à 110% du NRMSE du modèle de référence, la qualité est considérée comme **mauvaise** ❌.
            """)

        # Lister les dossiers sauvegardés
        st.divider()
        st.header("Téléchargement du Modèle")
        model_files = sorted(
            [d for d in os.listdir(dossier_modeles_entraines) if os.path.isdir(os.path.join(dossier_modeles_entraines, d))],
            reverse=True
        )

        if model_files:
            selected_model = st.selectbox("Sélectionnez le modèle à télécharger :", model_files)

            # Option pour renommer le modèle
            new_model_name = st.text_input("Renommer le modèle (laissez vide pour garder le nom actuel) :", value=selected_model)

            # Renommer le dossier si un nouveau nom est saisi
            if new_model_name and new_model_name != selected_model:
                old_path = os.path.join(dossier_modeles_entraines, selected_model)
                new_path = os.path.join(dossier_modeles_entraines, new_model_name)
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    st.success(f"Le modèle a été renommé en : {new_model_name}")
                    st.rerun()  # Recharger la page pour mettre à jour la liste des modèles
                else:
                    st.error(f"Un modèle avec le nom '{new_model_name}' existe déjà. Veuillez choisir un autre nom.")

            # Boutons pour télécharger et vider les modèles
            col1, col2 = st.columns([1, 1])
            with col1:
                model_path = os.path.join(dossier_modeles_entraines, selected_model)
                shutil.make_archive(model_path, 'zip', model_path)
                with open(f"{model_path}.zip", "rb") as f:
                    st.download_button(
                        label="Télécharger le modèle",
                        data=f,
                        file_name=f"{selected_model}.zip",
                        mime="application/zip"
                    )
            with col2:
                if st.button("Supprimer les modèles entraînés"):
                    for model_dir in os.listdir(dossier_modeles_entraines):
                        dir_path = os.path.join(dossier_modeles_entraines, model_dir)
                        if os.path.isdir(dir_path):
                            shutil.rmtree(dir_path)
                    st.success("Tous les modèles ont été supprimés.")
                    # Mettre à jour la liste déroulante après suppression
                    st.rerun()
        else:
            st.write("Aucun modèle disponible pour le téléchargement.")

if __name__ == "__main__":
    show()
