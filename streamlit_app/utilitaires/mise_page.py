import streamlit as st

# Fonction pour ameliorer l'affichage des éléments page statistique
def mise_forme_checkbox_radio():
    
    """ Applique du style CSS pour améliorer affichage des chckboxes et bouton radio sans impacter le menu """

    st.markdown(
    """
    <style>
    /* Réduire l'espace entre les checkboxes dans la page principale uniquement */
    div[data-testid="stCheckbox"] {
        margin-top: -5px !important;  /* Réduit l'espace entre les checkboxes */
        margin-bottom: -5px !important;
        display: flex;  /* Force l'alignement en ligne */
        align-items: center; /* Assure l'alignement vertical */
    }

    /* Ajustement spécifique de la troisième checkbox */
    div[data-testid="stCheckbox"]:last-of-type {
        margin-top: -10px !important;  /* Remonte la troisième checkbox */
    }

    /* Réduction de l'espace entre les boutons radio dans la page principale uniquement */
    div[data-testid="stRadio"] {
        margin-top: -38px !important;  /* Réduit l'espace entre les boutons radio */
        margin-bottom: -10px !important;
    }

    /* Restaurer l'espacement normal du menu dans la sidebar */
    div[data-testid="stSidebar"] div {
        margin-bottom: 0px !important;  /* Assure qu'il n'y a pas d'impact sur l'espacement */
    }

    /* Assurer que le contenu dans la sidebar reste avec son espacement normal */
    div[data-testid="stSidebarContent"] {
        padding-top: 0px !important;
        padding-bottom: 0px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
    )


