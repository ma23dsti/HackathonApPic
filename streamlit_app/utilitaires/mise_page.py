import streamlit as st

# Fonction pour ameliorer l'affichage des éléments page statistique
def mise_forme_checkbox_radio():
    
    """ Applique du style CSS pour améliorer affichage des checkboxes et bouton radio sans impacter le menu """

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

# Fonction pour réduire espace entre les titres stMarkdown et texte explicatif
def reduction_espace_titre_texte():
    st.markdown(
        """
    
        <style>
        /* Réduction de l'espace sous le titre principal */
        div[data-testid="stMarkdown"] h4 {
            margin-bottom: -15px !important;  /* Réduit  l’espace sous le titre */
        }

        /* Réduction de l'espace au-dessus du texte explicatif */
        p[style*="font-size: 14px; font-style: italic;"] {
            margin-top: -10px !important;  /* Remonte le texte explicatif */
            margin-bottom: -15px !important;  /* Supprime l’espace sous le texte explicatif */
            padding-bottom: 0px !important;
        }
        </style>
        """
        ,
        unsafe_allow_html=True
        )
    

# fonction pour grossir les bulles de valeurs du slider
def style_SliderThumbValue():
    st.markdown("""
        <style>
        /* Bulles de valeurs du slider */
        div[data-testid="stSliderThumbValue"] {
            font-weight: bold !important;
            font-size: 1.01em !important;
        }
        </style>
        """, unsafe_allow_html=True)    

# fonction pour mettre style des titres en bleu
def style_titre_bleu():
    st.markdown("""
        <style>
        h1 {
            color: #1870b8 !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
def afficher_bandeau_titre():
    st.markdown("""
    <style>
    h1 {
        background-color: #1870b8;
        color: white !important;
        padding: 1rem;
        border-radius: 6px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)