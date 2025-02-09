import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def show():
    st.title("Statistiques")
    st.write("Affichage des statistiques du modèle.")

    # Dummy statistics for testing
    stats = {
        "Précision": "95%",
        "Rappel": "90%",
        "F1-Score": "92%"
    }
    st.write(stats)

    # Dummy data for plotting
    epochs = np.arange(1, 11)
    accuracy = np.random.rand(10) * 10 + 90  # Random accuracy data
    loss = np.random.rand(10)  # Random loss data

    # Plotting accuracy
    fig, ax = plt.subplots()
    ax.plot(epochs, accuracy, label='Précision')
    ax.set_xlabel('Époques')
    ax.set_ylabel('Précision')
    ax.set_title('Précision au cours des époques')
    ax.legend()
    st.pyplot(fig)

    # Plotting loss
    fig, ax = plt.subplots()
    ax.plot(epochs, loss, label='Perte', color='red')
    ax.set_xlabel('Époques')
    ax.set_ylabel('Perte')
    ax.set_title('Perte au cours des époques')
    ax.legend()
    st.pyplot(fig)

    # Displaying data in an interactive table
    st.write("Tableau des données d'entraînement")
    data = {
        "Époques": epochs,
        "Précision": accuracy,
        "Perte": loss
    }
    df = pd.DataFrame(data)
    st.dataframe(df)  # Utilisation de st.dataframe pour afficher le tableau

if __name__ == "__main__":
    show()
