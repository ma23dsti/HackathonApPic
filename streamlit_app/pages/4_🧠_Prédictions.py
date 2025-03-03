import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from menu import display_menu
from dependency_manager import check_dependencies

display_menu()

def show():
    st.title("Prédictions")

    check_dependencies("Prédictions")

    # Générer des données aléatoires pour la démonstration
    input_data = pd.DataFrame({
        'value': np.random.randn(60).cumsum()  # Données cumulées pour simuler une série temporelle
    })

    # Générer des prédictions aléatoires
    predictions = np.random.randn(20).cumsum()  # Prédictions cumulées pour simuler une série temporelle

    # Créer un DataFrame pour les prédictions
    predictions_df = pd.DataFrame({
        'Index': range(len(input_data), len(input_data) + len(predictions)),
        'Predictions': predictions
    })

    # Stocker les données dans session_state pour les maintenir entre les exécutions
    if 'input_data' not in st.session_state:
        st.session_state.input_data = input_data
        st.session_state.predictions_df = predictions_df

    # Bouton pour faire les prédictions
    if st.button("Faire une prédiction"):
        # Afficher les prédictions sous forme de graphique
        plt.figure(figsize=(12, 6))  # Taille fixe pour la figure
        plt.plot(st.session_state.input_data.index, st.session_state.input_data['value'], label="Données d'entrée", color='blue')
        plt.plot(st.session_state.predictions_df['Index'], st.session_state.predictions_df['Predictions'], label="Prédictions", color='red')
        plt.axvline(x=len(st.session_state.input_data) - 1, color='black', linestyle='--')
        plt.xlabel("Index")
        plt.ylabel("Valeur")
        plt.title("Prédictions du modèle")
        plt.legend()
        st.pyplot(plt)

        st.session_state.valid_predictions = True

        # Ajouter une séparation
    st.markdown("---")

    # Afficher les prédictions dans un DataFrame horizontalement
    st.dataframe(st.session_state.predictions_df)

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
