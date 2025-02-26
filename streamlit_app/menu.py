import streamlit as st

def display_menu():
    # Vérifier si 'choix_modele' est initialisé dans st.session_state
    if 'choix_modele' not in st.session_state:
        st.session_state['choix_modele'] = None

    # Afficher le menu en fonction de l'état de 'choix_modele'
    if st.session_state.choix_modele == 0:
        st.sidebar.page_link("1_🏠_Accueil.py", label="Accueil", icon="🏠")
        st.sidebar.page_link("pages/2_📥_Dépot_et_Validation_des_Données.py", label="Dépot et validation des données", icon="📥")
        st.sidebar.page_link("pages/4_🧠_Prédictions.py", label="Prédicitons", icon="🧠")
        st.sidebar.page_link("pages/5_📊_Statistiques.py", label="Statistiques", icon="📊")
        st.sidebar.page_link("pages/6_ℹ️_À_propos.py", label="À propos", icon="ℹ️")
        st.sidebar.markdown("""---""")
        st.sidebar.write("**session_state pour debug :**")
        st.sidebar.write(st.session_state)

    elif st.session_state.choix_modele == 1:
        st.sidebar.page_link("1_🏠_Accueil.py", label="Accueil", icon="🏠")
        st.sidebar.page_link("pages/2_📥_Dépot_et_Validation_des_Données_2.py", label="Dépot et validation des données", icon="📥")
        st.sidebar.page_link("pages/3_🛠️_Entrainement_du_Modèle.py", label="Entraînement du modèle", icon="🛠️")
        st.sidebar.page_link("pages/4_🧠_Prédictions.py", label="Prédicitons", icon="🧠")
        st.sidebar.page_link("pages/5_📊_Statistiques.py", label="Statistiques", icon="📊")
        st.sidebar.page_link("pages/6_ℹ️_À_propos.py", label="À propos", icon="ℹ️")
        st.sidebar.markdown("""---""")
        st.sidebar.write("**session_state pour debug :**")
        st.sidebar.write(st.session_state)
    
    elif st.session_state.choix_modele == 2:
        st.sidebar.page_link("1_🏠_Accueil.py", label="Accueil", icon="🏠")
        st.sidebar.page_link("pages/2_📥_Dépot_et_Validation_des_Données_3.py", label="Dépot du modèle et des données", icon="📥")
        st.sidebar.page_link("pages/4_🧠_Prédictions.py", label="Prédicitons", icon="🧠")
        st.sidebar.page_link("pages/5_📊_Statistiques.py", label="Statistiques", icon="📊")
        st.sidebar.page_link("pages/6_ℹ️_À_propos.py", label="À propos", icon="ℹ️")
        st.sidebar.markdown("""---""")
        st.sidebar.write("**session_state pour debug :**")
        st.sidebar.write(st.session_state)

    else:
        # Cas par défaut : afficher uniquement "Accueil"
        st.sidebar.page_link("1_🏠_Accueil.py", label="Accueil", icon="🏠")
        