import streamlit as st
st.set_page_config(layout="wide")  # Active le mode large

from datetime import timedelta
from menu import display_menu

from dependency_manager import check_dependencies
from utilitaires.Chargement import chargement_donnees
from utilitaires.Visualisation_Stat import palette_defaut, creation_graphique, creation_tableau
from utilitaires.Selection_Parametre import selection_parametre, selection_plage_date, selection_titre, selection_variable_filtre, selection_donnees_format_export
from utilitaires.Export_Resultat import export_data_zip
from streamlit_extras.add_vertical_space import add_vertical_space
from utilitaires.mise_page import mise_forme_checkbox_radio
import plotly.express as px
import tempfile


display_menu()

def show():
    st.title("Statistiques")

    check_dependencies("Statistiques")

    # en attendant d'avoir les vraies valeurs chargement de données pour faire des essais - Charger les données au démarrage
    df_entrees_prevision, donnees_kpi, col_temps, liste_modeles_id,nb_modele,id_modele_moyen,id_modele_entree,liste_unite,mesure_format, var_id, var_val,min_date,max_date,min_date_entree, max_date_entree, min_date_prevision, max_date_prevision=chargement_donnees()
    
    
    st.markdown(
    """
   
    <style>
    /* Réduction de l'espace sous le titre principal */
    div[data-testid="stMarkdown"] h4 {
        margin-bottom: -15px !important;  /* Réduit encore plus l’espace sous le titre */
    }

    /* Réduction de l'espace au-dessus du texte explicatif */
    p[style*="font-size: 14px; font-style: italic;"] {
        margin-top: -10px !important;  /* Remonte encore plus le texte explicatif */
        margin-bottom: -15px !important;  /* Supprime l’espace sous le texte explicatif */
        padding-bottom: 0px !important;
    }
    </style>
    """
    ,
    unsafe_allow_html=True
    )
   


   
    st.markdown("#### 🔧 Sélection des Paramètres à afficher") 
    st.markdown("<p style='font-size:14px; font-style:italic;'>Le graphique et le tableau des métriques se mettront à jour dynamiquement en fonction de vos sélections.</p>", unsafe_allow_html=True)
    
    # Initialisation des paramétres
    st.session_state.setdefault("affichage_modele_entree", True)
    st.session_state.setdefault("choix_temps", "Temps horaire")
    st.session_state.setdefault("affichage_ensemble_prediction", True)
    st.session_state.setdefault("affichage_moyenne_prediction", False)
    st.session_state.setdefault("selection_date", (min_date, max_date))
    st.session_state.setdefault("choix_unite", liste_unite[0])
    #st.session_state.setdefault("export_donnees",True)

    selection_date = st.session_state["selection_date"]
    
    mise_forme_checkbox_radio()

    # Selection des parametres et variables
    affichage_modele_entree, choix_temps, affichage_ensemble_prediction, affichage_moyenne_prediction, choix_unite= selection_parametre(liste_unite,nb_modele,min_date,max_date, selection_date)
    
    #création d'une liste pour simplifier la gestion de la selection des affichages
    selection_options = {"affichage_modele_entree": st.session_state.affichage_modele_entree,
                         "affichage_ensemble_prediction": st.session_state.affichage_ensemble_prediction,
                         "affichage_moyenne_prediction": st.session_state.affichage_moyenne_prediction}
    
    #st.session_state["selection_date"] = {"debut": debut_date, "fin": fin_date}

    
   
    #Filtre sur l'axe des temps
    st.markdown("📅 Fenêtre temporelle:")


    st.markdown(
    f"""
    <div style="background-color:#f5f5f5; padding:10px; border-radius:5px;">
        <em style="color:#333333; font-size:14px;">
        ⚠️ Attention :<br>
        Les données d’entrée sont disponibles entre <strong>{min_date_entree} et {max_date_entree}</strong>,
        et les prévisions entre <strong>{min_date_prevision} et {max_date_prevision}</strong>.<br>
        Si vous sélectionnez une période excluant certains intervalles,les données concernées ne seront pas affichées même si elles sont sélectionnées.
        </em>
    </div>
    """,
    unsafe_allow_html=True
    )

    couleur_slider = "#00FF00" 
        #if not mode_daltonien else "#FFD700"

    st.markdown(
        f"""
        <style>
        .stSlider [data-baseweb="slider"] > div > div span {{color: {couleur_slider} !important;}}
        </style>
        """,
        unsafe_allow_html=True
    )
    #debut_session = st.session_state["selection_date"]["debut"]
    #fin_session = st.session_state["selection_date"]["fin"]

    debut_date, fin_date = selection_plage_date(min_date, max_date)



    if (debut_date != st.session_state["selection_date"][0]) or (fin_date != st.session_state["selection_date"][1]):
        st.session_state["selection_date"] = (debut_date, fin_date)

     
    
    # Données filtres suite à la sélection des parametres
    df_entrees_prevision_selection, df_kpi_selection, liste_donnees_filtre=selection_variable_filtre(id_modele_entree, selection_options, selection_date, df_entrees_prevision, donnees_kpi, liste_modeles_id, var_id, id_modele_moyen)


    # titre_graphe, label_x, label_y, df_kpi_selection, df_entrees_prevision_selection, liste_donnees_filtre=selection_variable(id_historique, choix_temps, affichage_historique, affichage_moyenne_prediction, affichage_ensemble_prediction, df_entrees_prevision, donnees_kpi, liste_modeles_id,mesure_format,choix_unite, var_id,id_modele_moyen)
    titre_graphe, label_x, label_y=selection_titre(selection_options, choix_temps, mesure_format, choix_unite)
    
    add_vertical_space(2) # ajout espace
    st.markdown("#### 📈 Affichage des Résultats")

    # Mise en page en deux colonnes
    col1, col2 = st.columns([2, 5])  # Col1 = 1/4 de la page, Col2 = 3/4
    with col1:

        st.markdown("#####  Rappel des Métriques")
        tab=creation_tableau (df_kpi_selection)
        st.plotly_chart(tab, use_container_width=True)
        #st.plotly_chart(tab)
        #st.markdown("<div style='margin-top: -50px;'></div>", unsafe_allow_html=True)

        
    with col2:
        st.markdown(f" ##### {titre_graphe}")
        fig=creation_graphique(df_entrees_prevision_selection, palette_defaut, liste_donnees_filtre, var_id,choix_temps,var_val, label_x,label_y)
        st.pyplot(fig)
        st.markdown("<div style='margin-top: -50px;'></div>", unsafe_allow_html=True)
        

    #add_vertical_space(2)

    #Export des données selectionnées
    st.markdown("#### Sélection des paramètres d'export")
    st.markdown("<p style='font-size:14px; font-style:italic;'>Sélectionner les données et format à exporter puis cliquer sur le bouton pour les exporter en fichier zip.</p>", unsafe_allow_html=True)
    st.markdown(" ")
    


    export_options=selection_donnees_format_export()
    

    # Initialisation des états zip
    if "zip_ready" not in st.session_state:
        st.session_state.zip_ready = False
    if "zip_path" not in st.session_state:
        st.session_state.zip_path = None  # Chemin du fichier ZIP temporaire

    # Bouton pour générer le fichier ZIP
    if st.button("📝 Générer le fichier ZIP"):
        st.session_state.zip_ready = False  # Réinitialisation
        with st.spinner("⏳ Préparation du fichier ZIP en cours... Veuillez patienter."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                zip_file_path = tmp_zip.name  # Chemin du fichier temporaire qui est généré
                # récupération du fichier via la fonction (on stocke en memoire le chemin pas le fichier)
                st.session_state.zip_path =export_data_zip(df_entrees_prevision_selection, df_kpi_selection, export_options, fig, tab, titre_graphe, zip_file_path)
                
                st.session_state.zip_ready = True  # Indique que le fichier est prêt
            
                
    # Affichage du bouton de téléchargement quand le fichier est prêt
    if st.session_state.zip_ready and st.session_state.zip_path:
        st.success("📦 Le fichier ZIP est prêt ! Vous pouvez le télécharger.")
        with open(st.session_state.zip_path, "rb") as zip_file:
            st.download_button(
                label="⬇️ Télécharger le fichier ZIP",
                data=zip_file,
                file_name="export_resultats.zip",
                mime="application/zip"
            )

    # Réinitialisation du ZIP si l'utilisateur change une sélection
    if any(st.session_state[key] for key in {**export_options["formats"], **export_options["donnees"]}):
        st.session_state.zip_ready = False  # Force la régénération
        st.session_state.zip_path = None  # Supprime le chemin du fichier

    



    st.session_state.valid_statistiques = True

if __name__ == "__main__":
    show()
