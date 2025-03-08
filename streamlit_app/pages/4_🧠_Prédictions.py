import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from menu import display_menu
from dependency_manager import check_dependencies
from utilitaires.Prediction import predire_le_traffic

display_menu()

preprocessing_dir = "streamlit_app/static/donnees/donnees_preprocessees/"
dossier_donnees_pour_entrainement = preprocessing_dir + "donnees_a_la_volee/"


def show():
    st.title("Prédictions")

    check_dependencies("Prédictions")

    # Vérifier si les données sont déjà disponibles dans la session
    #if 'prediction_data' not in st.session_state:
    #    st.error("Aucune donnée validée. Veuillez d'abord valider les données sur la première page.")
    #    return

    # S'assurer que les données sont sous forme de DataFrame avec une colonne 'value'
    if not isinstance(st.session_state.prediction_data, pd.DataFrame) or 'value' not in st.session_state.prediction_data.columns:
    #if not isinstance(st.session_state.prediction_data, pd.DataFrame) or 'value' not in st.session_state.prediction_data.columns:
        st.session_state.prediction_data = pd.DataFrame({'value': st.session_state.prediction_data.values.flatten()})
        st.session_state.prediction_data.index = range(1, len(st.session_state.prediction_data) + 1)

    # Bouton pour faire les prédictions
    if st.button("Effectuer la prédiction"):

        if 'predictions_df' not in st.session_state or not st.session_state.prediction_effectuee:
            # Flatten and reshape to (1, 60) for model prediction
            prediction_data_reshaped = np.array(st.session_state.prediction_data).flatten().reshape(1, -1)
            # Ensure we have 60 features
            if prediction_data_reshaped.shape[1] != 60:
                st.error(f"Erreur: Le modèle attend 60 colonnes, mais {prediction_data_reshaped.shape[1]} ont été détectées.")
                return
            predictions = predire_le_traffic(prediction_data_reshaped)
            st.write("Prédiction terminée avec succès")

            predictions = np.array(predictions).flatten()
            # Check lengths to prevent errors
            if len(predictions) != 5:  # Expected next 5 values in time series
                st.error(f"Erreur: Le modèle a généré {len(predictions)} valeurs, mais 5 étaient attendues.")
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
        st.write("### Prédictions générées:")
        st.write(st.session_state.predictions_df)

        # Afficher les prédictions sous forme de graphique
        plt.figure(figsize=(12, 6))
        plt.plot(st.session_state.prediction_data.index, st.session_state.prediction_data['value'], label="Données d'entrée", color='blue')
        plt.plot(st.session_state.predictions_df['Index'], st.session_state.predictions_df['Predictions'], label="Prédictions", color='red')
        plt.axvline(x=len(st.session_state.prediction_data), color='black', linestyle='--')
        plt.xlabel("Index")
        plt.ylabel("Valeur")
        plt.title("Prédictions du modèle")
        plt.legend()
        st.pyplot(plt)

        # Ajouter une séparation
        st.markdown("---")

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