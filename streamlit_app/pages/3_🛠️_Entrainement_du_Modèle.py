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

            # Définir un nombre aléatoire d'étapes entre 5 et 10
            ##st.session_state.total_steps = np.random.randint(5, 11)
            ##progress_text = f"Entraînement en cours... {st.session_state.total_steps} étapes au total."

            # Créer une progress bar
            ##progress_bar = st.progress(0, text=progress_text)

            # Dummy training logic (à remplacer par l'appel au backend)
            ##for i in range(1, st.session_state.total_steps + 1):
                ##progress_bar.progress(i / st.session_state.total_steps, text=f"Étape {i}/{st.session_state.total_steps} : {progress_text}")
                ##time.sleep(1)  # Simuler le temps de traitement

            ##progress_bar.empty()  # Supprimer la progress bar après la fin de l'entraînement
            st.success("Entraînement terminé.")

            # Après un (ré)entrainement de modèle, l'historique des resultats aura besoin d'être recalculé lors de la prochaine prédiction.
            if 'prediction_historique_recalculee' in st.session_state:
                st.session_state.prediction_historique_recalculee = False

            # Générer une valeur aléatoire pour le NRMSE dans une plage décente
            ##st.session_state.nrmse_value = np.random.uniform(0.05, 0.25)  # Plage décente pour un modèle de qualité moyenne

            # Créer un modèle dummy avec PyTorch
            class DummyModel(nn.Module):
                def __init__(self):
                    super(DummyModel, self).__init__()
                    self.layer = nn.Linear(10, 1)

                def forward(self, x):
                    return self.layer(x)

            dummy_model = DummyModel()
            model_filename = f"dummy_model_{st.session_state.nrmse_value:.4f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            model_path = os.path.join("streamlit_app/static/modeles/modeles_entraines", model_filename)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(dummy_model.state_dict(), model_path)

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

        # Lister les modèles sauvegardés
        st.divider()
        st.header("Téléchargement du Modèle")
        model_files = [f.replace(".pth", "") for f in os.listdir("streamlit_app/static/modeles/modeles_entraines") if f.endswith(".pth")]

        if model_files:
            selected_model = st.selectbox("Sélectionnez le modèle à télécharger :", model_files)

            # Option pour renommer le modèle
            new_model_name = st.text_input("Renommer le modèle (laissez vide pour garder le nom actuel) :", value=selected_model)

            # Boutons pour télécharger et vider les modèles
            col1, col2 = st.columns([1, 1])
            with col1:
                model_path = os.path.join("streamlit_app/static/modeles/modeles_entraines", f"{selected_model}.pth")
                with open(model_path, "rb") as f:
                    st.download_button(
                        label="Télécharger le modèle",
                        data=f,
                        file_name=f"{new_model_name if new_model_name else selected_model}.pth",
                        mime="application/octet-stream"
                    )
            with col2:
                if st.button("Supprimer les modèles entraînés"):
                    for model_file in os.listdir("streamlit_app/static/modeles/modeles_entraines"):
                        if model_file.endswith(".pth"):
                            os.remove(os.path.join("streamlit_app/static/modeles/modeles_entraines", model_file))
                    st.success("Tous les modèles ont été supprimés.")
                    # Mettre à jour la liste déroulante après suppression
                    st.rerun()
        else:
            st.write("Aucun modèle disponible pour le téléchargement.")

if __name__ == "__main__":
    show()
