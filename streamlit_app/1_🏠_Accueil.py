import streamlit as st
from menu import display_menu

st.set_page_config(page_title="Plateforme de Prédiction de Trafic Réseau", page_icon="🚦", layout="wide")

def show():
    st.title("Plateforme de Prédiction de Trafic Réseau")
    st.header("""
    Bienvenue sur la plateforme de prédiction de trafic réseau !
    Utilisez la barre latérale pour naviguer entre les pages.
    """, divider=True)

    # Initialiser st.session_state
    if 'choix_modele' not in st.session_state:
        st.session_state['choix_modele'] = None
    if 'taille_fenetre' not in st.session_state:
        st.session_state['taille_fenetre'] = None
    if 'nombre_predictions' not in st.session_state:
        st.session_state['nombre_predictions'] = None

    # Question pour l'utilisateur
    choix_modele_options = {
        "Prédire avec le modèle pré-chargé": 0,
        "Entrainer un modèle et faire une prédiction": 1,
        "Prédire avec un modèle externe à charger sur la plateforme": 2
    }

    st.markdown(
    """
        <style>
    div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 26px;
    }
        </style>
    """, unsafe_allow_html=True)

    # Utiliser st.radio pour détecter les changements
    choix_modele = st.radio(
        "**Choisissez une option :**",
        options=list(choix_modele_options.keys()),
        index=None
    )

    # Options avec alias pour la taille de la fenêtre
    # Ajouter les autres options dans ce dictionnaire
    taille_fenetre_options = {
        "60 secondes": 60,
        "90 secondes": 90 # valeur test pour l'affichage
    }
    taille_fenetre = st.selectbox("Taille de la fenêtre", options=list(taille_fenetre_options.keys()), index=None, placeholder="Choisis une option")

    # Options avec alias pour le nombre de prédictions
    # Ajouter les autres options dans ce dictionnaire
    nombre_predictions_options = {
        "5 prédictions": 5,
        "20 prédictions": 20 # valeur test pour l'affichage
    }
    nombre_predictions = st.selectbox("Nombre de prédictions", options=list(nombre_predictions_options.keys()), index=None, placeholder="Choisis une option")

    # Bouton pour valider les choix et mettre à jour le menu dans la sidebar
    if st.button("Valider les choix"):
        if choix_modele is None:
            st.error("Veuillez choisir une option valide pour le modèle.")
        elif taille_fenetre is None:
            st.error("Veuillez choisir une option valide pour la taille de la fenêtre.")
        elif nombre_predictions is None:
            st.error("Veuillez choisir une option valide pour le nombre de prédictions.")
        else:
            st.session_state['choix_modele'] = choix_modele_options[choix_modele]
            st.session_state['taille_fenetre'] = taille_fenetre_options[taille_fenetre]
            st.session_state['nombre_predictions'] = nombre_predictions_options[nombre_predictions]
            st.rerun()

    st.markdown("""---""")

if __name__ == "__main__":

    # Initialiser st.session_state sinon display_menu() ne fonctionnera pas
    if 'choix_modele' not in st.session_state:
        st.session_state['choix_modele'] = None
    if 'taille_fenetre' not in st.session_state:
        st.session_state['taille_fenetre'] = None
    if 'nombre_predictions' not in st.session_state:
        st.session_state['nombre_predictions'] = None

    with st.sidebar:
        if st.session_state.choix_modele is None:
            st.sidebar.page_link("1_🏠_Accueil.py", label="Accueil", icon="🏠")
        else:
            display_menu()
    show()
