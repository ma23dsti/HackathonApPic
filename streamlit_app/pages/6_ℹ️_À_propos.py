import streamlit as st
from menu import display_menu

display_menu()

def show():
    """
    Affiche la page "À propos".

    Cette fonction affiche le titre et le contenu de la page "À propos" de la plateforme.
    Elle fournit des informations sur le contexte de développement de la plateforme.

    Parameters:
    None

    Returns:
    None
    """
    st.title("À propos")
    st.write("""
    Cette plateforme a été développée dans le cadre d'un hackathon pour prédire les valeurs des 5 prochaines secondes à partir de 60 secondes de données.
    """)

if __name__ == "__main__":
    show()
