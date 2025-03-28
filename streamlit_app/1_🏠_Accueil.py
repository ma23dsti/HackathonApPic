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

    # Options pour la taille de la fenêtre et le nombre de prédictions
    options_combinees = {
        "5 prédictions / 60 secondes": (5, 60),
        "30 prédictions / 360 secondes": (30, 360),
        "60 prédictions / 720 secondes": (60, 720),
        "300 prédictions / 3600 secondes": (300, 3600)
    }

    # Afficher les options dans une seule selectbox
    option_select = st.selectbox(
        "Choisissez le nombre de prédictions et la taille de la fenêtre",
        options=list(options_combinees.keys()),
        index=None,
        placeholder="Choisissez une option"
    )

    # Ajouter les autres options dans ce dictionnaire
    unite_mesure_options = [
        "Octets",
        "Bits/s"
    ]
    unite_mesure = st.selectbox("Unité de mesure", options=unite_mesure_options, index=None, placeholder="Choisis une option")


    # Bouton pour valider les choix et mettre à jour le menu dans la sidebar
    if st.button("Valider les choix"):
        if choix_modele is None:
            st.error("Veuillez choisir une option valide pour le modèle.")
        elif unite_mesure is None:
            st.error("Veuillez choisir une unité de mesure.")
        elif option_select is None:
            st.error("Veuillez choisir une option valide pour le nombre de prédictions et la taille de la fenêtre.")
        else:
            st.session_state['choix_modele'] = choix_modele_options[choix_modele]
            st.session_state['taille_fenetre'] = options_combinees[option_select][1]
            st.session_state['nombre_predictions'] = options_combinees[option_select][0]
            st.session_state['unite_mesure'] = unite_mesure
            st.session_state.valid_acceuil = True
            st.rerun()
        
    # Message de validation pour l'utilisateur afin de passer à l'étape suivante
    if st.session_state.valid_acceuil:
        st.success("Choix validé avec succès ! Vous pouvez passer à l'étape suivante.")

    st.markdown("""---""")

if __name__ == "__main__":

    # Initialiser st.session_state sinon display_menu() ne fonctionnera pas
    # Liste des clés à initialiser
    key_user_choices = ['choix_modele', 'taille_fenetre', 'nombre_predictions', 'unite_mesure']
    keys_to_initialize = ['valid_acceuil', 'valid_depot_donnees', 'valid_entrainement', 'valid_predictions', 'valid_statistiques']

    # Initialisation des clés dans st.session_state
    for key in key_user_choices:
        if key not in st.session_state:
            st.session_state[key] = None

    for key in keys_to_initialize:
        if key not in st.session_state:
            st.session_state[key] = False

    with st.sidebar:
        if st.session_state.choix_modele is None:
            st.sidebar.page_link("1_🏠_Accueil.py", label="Accueil", icon="🏠")
            st.sidebar.markdown("""---""")
            st.sidebar.write("**session_state pour debug :**")
            st.sidebar.write(st.session_state)
        else:
            display_menu()
    show()
