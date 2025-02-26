import streamlit as st
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from menu import display_menu

display_menu()

def show():
    st.title("Prédictions")

    # Générer des données aléatoires pour la démonstration
    np.random.seed(42)  # Pour la reproductibilité
    input_data = pd.DataFrame({
        'value': np.random.randn(60).cumsum()  # Données cumulées pour simuler une série temporelle
    })

    # Générer des prédictions aléatoires
    predictions = np.random.randn(20).cumsum()  # Prédictions cumulées pour simuler une série temporelle

    # Bouton pour faire les prédictions
    if st.button("Faire une prédiction"):
        # Afficher les prédictions sous forme de graphique
        plt.figure(figsize=(24, 12))
        plt.plot(range(len(input_data)), input_data['value'], label="Données d'entrée", color='blue')
        plt.plot(range(len(input_data), len(input_data) + len(predictions)), predictions, label="Prédictions", color='red')
        plt.axvline(x=len(input_data) - 1, color='black', linestyle='--')
        plt.xlabel("Index")
        plt.ylabel("Valeur")
        plt.title("Prédictions du modèle")
        plt.legend()
        st.pyplot(plt)

if __name__ == "__main__":
    show()
