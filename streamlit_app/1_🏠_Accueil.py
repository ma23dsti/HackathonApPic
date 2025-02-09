import streamlit as st

st.set_page_config(page_title="Plateforme de Prédiction de Trafic Réseau", page_icon="🚦", layout="wide")

def show():
    st.title("Plateforme de Prédiction de Trafic Réseau")
    st.write("""
    Bienvenue sur la plateforme de prédiction de trafic réseau ! 
    Utilisez la barre latérale pour naviguer entre les pages.
    """)

if __name__ == "__main__":
    show()
