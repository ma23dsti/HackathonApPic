import streamlit as st
from menu import display_menu
from utilitaires.mise_page import afficher_bandeau_titre

st.set_page_config(page_title="Plateforme de Prédiction de Trafic Réseau", page_icon="🚦", layout="wide")

def show():
    """
    Affiche la page d'accueil de la plateforme de prédiction de trafic réseau.

    Cette fonction initialise les états de session nécessaires, affiche les options pour l'utilisateur,
    et met à jour les états de session en fonction des choix de l'utilisateur. Elle fournit également
    des messages de validation pour guider l'utilisateur à travers les étapes de configuration.

    Parameters:
    None

    Returns:
    None
    """
    afficher_bandeau_titre()
    st.title("Plateforme de Prédiction de Trafic Réseau")
    st.header("""
    Bienvenue sur la plateforme de prédiction de trafic réseau !
    Utilisez la barre latérale pour naviguer entre les pages.
    """, divider=True)

    # Initialiser st.session_state
    if 'choix_modele' not in st.session_state:
        st.session_state['choix_modele'] = None
    if 'taille_fenetre_observee' not in st.session_state:
        st.session_state['taille_fenetre_observee'] = None
    if 'horizon_predictions' not in st.session_state:
        st.session_state['horizon_predictions'] = None
    if 'premiere_prediction_seule' not in st.session_state:
        st.session_state['premiere_prediction_seule'] = None
    if 'prediction_avec_model_charge' not in st.session_state:
        st.session_state['prediction_avec_model_charge'] = None
    # Reset model_charge when on the home page
    if "model_charge" in st.session_state:
        st.session_state.model_charge = None
    # Set the on_homepage flag to reset the training plot
    st.session_state["on_homepage"] = True

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

    # Structure des modèles par défaut
    taille_fenetre_a_predire_mapping = {
        "taille_fenetre_a_predire": {
            1: {
                "taille_fenetre_observee": 12,
                "taille_pas_glissant_train": 13,
                "taille_pas_glissant_valid": 13
            },
            5: {
                "taille_fenetre_observee": 60,
                "taille_pas_glissant_train": 13,
                "taille_pas_glissant_valid": 65
            },
            30: {
                "taille_fenetre_observee": 300,
                "taille_pas_glissant_train": 13,
                "taille_pas_glissant_valid": 65
            },
            60: {
                "taille_fenetre_observee": 400,
                "taille_pas_glissant_train": 13,
                "taille_pas_glissant_valid": 65
            },
            300: {
                "taille_fenetre_observee": 500,
                "taille_pas_glissant_train": 13,
                "taille_pas_glissant_valid": 65
            }
        }
    }
    if 'taille_fenetre_a_predire_mapping' not in st.session_state:
        st.session_state['taille_fenetre_a_predire_mapping'] = taille_fenetre_a_predire_mapping

    # Options pour la taille de la fenêtre et le nombre de prédictions
    # Extraire les clés de "taille_fenetre_a_predire"
    cles = list(taille_fenetre_a_predire_mapping["taille_fenetre_a_predire"].keys())

    options_combinees = {
        "{} point à prédire / {} observations".format(
    cles[0], taille_fenetre_a_predire_mapping["taille_fenetre_a_predire"][cles[0]]["taille_fenetre_observee"]
): (cles[0], taille_fenetre_a_predire_mapping["taille_fenetre_a_predire"][cles[0]].values()),
        "{} points à prédire / {} observations".format(
    cles[1], taille_fenetre_a_predire_mapping["taille_fenetre_a_predire"][cles[1]]["taille_fenetre_observee"]
): (cles[1], taille_fenetre_a_predire_mapping["taille_fenetre_a_predire"][cles[1]].values()),
        "{} points à prédire / {} observations".format(
    cles[2], taille_fenetre_a_predire_mapping["taille_fenetre_a_predire"][cles[2]]["taille_fenetre_observee"]
): (cles[2], taille_fenetre_a_predire_mapping["taille_fenetre_a_predire"][cles[2]].values()),
        "{} points à prédire / {} observations".format(
    cles[3], taille_fenetre_a_predire_mapping["taille_fenetre_a_predire"][cles[3]]["taille_fenetre_observee"]
): (cles[3], taille_fenetre_a_predire_mapping["taille_fenetre_a_predire"][cles[3]].values()),
        "{} points à prédire / {} observations".format(
    cles[4], taille_fenetre_a_predire_mapping["taille_fenetre_a_predire"][cles[4]]["taille_fenetre_observee"]
): (cles[4], taille_fenetre_a_predire_mapping["taille_fenetre_a_predire"][cles[4]].values())
    }

    # Afficher les options dans une seule selectbox
    option_select = st.selectbox(
        "Choisissez le nombre de points à prédire et la taille de la fenêtre observée",
        options=list(options_combinees.keys()),
        index=None,
        placeholder="Choisissez une option"
    )

    # Ajouter les autres options dans ce dictionnaire
    unite_mesure_options = [
        "Bits/s",
        "Octets/s"
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
            st.session_state['taille_fenetre_observee'] = list(options_combinees[option_select][1])[0]
            st.session_state['horizon_predictions'] = options_combinees[option_select][0]
            st.session_state['unite_mesure'] = unite_mesure
            st.session_state['sliding_window_train'] = st.session_state.taille_fenetre_a_predire_mapping["taille_fenetre_a_predire"][st.session_state.horizon_predictions]["taille_pas_glissant_train"]
            st.session_state['sliding_window_valid'] = st.session_state.taille_fenetre_a_predire_mapping["taille_fenetre_a_predire"][st.session_state.horizon_predictions]["taille_pas_glissant_valid"]
            st.session_state['premiere_prediction_seule'] = True
            
            st.session_state.valid_acceuil = True
            st.rerun()
        
    # Message de validation pour l'utilisateur afin de passer à l'étape suivante
    if st.session_state.valid_acceuil:
        st.success("✅ Choix validé avec succès ! Vous pouvez passer à l'étape suivante.")

    st.markdown("""---""")

if __name__ == "__main__":

    # Initialiser st.session_state sinon display_menu() ne fonctionnera pas
    # Liste des clés à initialiser
    key_user_choices = ['choix_modele', 'taille_fenetre_observee', 'horizon_predictions', 'unite_mesure']
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
            #st.sidebar.write("**session_state pour debug :**")
            #st.sidebar.write(st.session_state)
        else:
            display_menu()
    show()
