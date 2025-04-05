import streamlit as st
from menu import display_menu

display_menu()

def show():
    st.title("À propos")
    st.write("""
    Cette plateforme a été développée dans le cadre d'un hackathon pour prédire les valeurs des 1, 5, 30, 60 ou 300 prochaines secondes à partir de 12, 60, 300, 400 ou 500 secondes de données.
    """)

if __name__ == "__main__":
    show()
