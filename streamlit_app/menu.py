import streamlit as st

# Fonction de dialogue pour la confirmation
@st.dialog("Confirmer le retour à l'accueil")
def confirm_reset():
    st.write('''Êtes-vous sûr de vouloir retourner à l'accueil ? Toutes les données non sauvegardées seront perdues.
              \n Pour reprendre là où vous en étiez, fermez ce pop-up.''')
    if st.button("Retourner à l'accueil"):
        st.session_state.clear()
        st.switch_page("1_🏠_Accueil.py")

def display_menu():
    # Vérifier si 'choix_modele' est initialisé dans st.session_state
    if 'choix_modele' not in st.session_state:
        st.session_state['choix_modele'] = None

    # Ajouter un bouton personnalisé pour retour à l'accueil avec confirmation
    if st.sidebar.button("🏠 Accueil", key="home_button"):
        confirm_reset()

    # Plus joli que le bouton mais ca marche pas comme je le souhaite et pas encore trouvé de solution
    # if st.sidebar.page_link("1_🏠_Accueil.py", label="Accueil", icon="🏠"):
    #     confirm_reset()


    # Afficher le menu en fonction de l'état de 'choix_modele'
    if st.session_state.choix_modele == 0:
        st.sidebar.page_link("pages/2_📥_Dépot_et_Validation_des_Données.py", label="Dépot et validation des données", icon="📥")
        st.sidebar.page_link("pages/4_🧠_Prédictions.py", label="Prédicitons", icon="🧠")
        st.sidebar.page_link("pages/5_📊_Statistiques.py", label="Statistiques", icon="📊")
        st.sidebar.page_link("pages/6_ℹ️_À_propos.py", label="À propos", icon="ℹ️")


    elif st.session_state.choix_modele == 1:
        st.sidebar.page_link("pages/2_📥_Dépot_et_Validation_des_Données_2.py", label="Dépot et validation des données", icon="📥")
        st.sidebar.page_link("pages/3_🛠️_Entrainement_du_Modèle.py", label="Entraînement du modèle", icon="🛠️")
        st.sidebar.page_link("pages/4_🧠_Prédictions.py", label="Prédicitons", icon="🧠")
        st.sidebar.page_link("pages/5_📊_Statistiques.py", label="Statistiques", icon="📊")
        st.sidebar.page_link("pages/6_ℹ️_À_propos.py", label="À propos", icon="ℹ️")

    elif st.session_state.choix_modele == 2:
        st.sidebar.page_link("pages/2_📥_Dépot_et_Validation_des_Données_3.py", label="Dépot du modèle et des données", icon="📥")
        st.sidebar.page_link("pages/4_🧠_Prédictions.py", label="Prédicitons", icon="🧠")
        st.sidebar.page_link("pages/5_📊_Statistiques.py", label="Statistiques", icon="📊")
        st.sidebar.page_link("pages/6_ℹ️_À_propos.py", label="À propos", icon="ℹ️")

    else:
        # Cas par défaut : afficher uniquement "Accueil"
        st.sidebar.markdown("""---""")
        st.sidebar.write("**session_state pour debug :**")
        st.sidebar.write(st.session_state)
    
    st.sidebar.markdown("""---""")

    # Dictionnaire pour les descriptions des choix de modèle
    choix_modele_descriptions = {
        0: "Prédire avec le modèle pré-chargé",
        1: "Entraîner un modèle et faire une prédiction",
        2: "Prédire avec un modèle externe à charger sur la plateforme"
    }

    # Afficher les étapes complétées
  
    # Afficher les choix de l'utilisateur et l'état de la session
    st.sidebar.markdown("### Paramètres sélectionnés pour la session")

    # Liste des choix des paramètres de l'utilisateur
    user_choices = []

    # Afficher le choix du modèle
    if st.session_state.choix_modele is not None:
        user_choices.append(f"**Option sélectionnée :** *{choix_modele_descriptions.get(st.session_state.choix_modele, 'Aucune')}*")

    # Afficher la taille de la fenêtre et le nombre de prédictions
    if 'taille_fenetre' in st.session_state:
        user_choices.append(f"**Taille de la fenêtre :** *{st.session_state.taille_fenetre}*")
    if 'nombre_predictions' in st.session_state:
        user_choices.append(f"**Nombre de prédictions :** *{st.session_state.nombre_predictions}*")

    st.sidebar.markdown("\n".join(f"- {choice}" for choice in user_choices))

    st.sidebar.markdown("""---""")

    # Afficher les étapes complétées sous forme de liste
    st.sidebar.markdown("### Étapes Complétées")
    completed_steps = []
    if st.session_state.get('valid_acceuil', False):
        completed_steps.append("✅ Accueil")
    if st.session_state.get('valid_depot_donnees', False):
        completed_steps.append("✅ Dépôt et Validation des Données")
    if st.session_state.get('valid_entrainement', False):
        completed_steps.append("✅ Entraînement du Modèle")
    if st.session_state.get('valid_predictions', False):
        completed_steps.append("✅ Prédictions")
    if st.session_state.get('valid_statistiques', False):
        completed_steps.append("✅ Statistiques")

    st.sidebar.markdown("\n".join(f"- {step}" for step in completed_steps))


    
    st.sidebar.markdown("""---""")
    st.sidebar.write("**session_state pour debug :**")
    st.sidebar.write(st.session_state)
