import streamlit as st
import numpy as np
import pandas as pd

def show():
    st.title("Dépot et Validation des Données")
    st.write("Veuillez entrer les données pour les 60 dernières secondes:")

    # Espace de dépôt des données
    uploaded_file = st.file_uploader("Déposez vos fichiers ici", type=["csv", "txt"])
    
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.write("Données d'entrée:", input_data)
    else:
        # Dummy input data for testing
        input_data = np.random.rand(1, 60)
        st.write("Données d'entrée:", input_data)

    if st.button("Valider"):
        # Dummy validation logic
        if np.any(np.isnan(input_data)):
            st.write("Erreur: Les données contiennent des valeurs manquantes.")
        elif np.any(input_data < 0):
            st.write("Erreur: Les données contiennent des valeurs négatives.")
        else:
            st.write("Les données sont valides.")

if __name__ == "__main__":
    show()
