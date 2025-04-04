import streamlit as st
st.set_page_config(layout="wide")  # Active le mode large

from datetime import timedelta
from menu import display_menu

from dependency_manager import check_dependencies
from utilitaires.Stat_Chargement import charger_fichier_json, extraire_donnees_entree, extraire_donnees_prediction_et_kpi, fusionner_et_convertir, obtenir_info_metadata
from utilitaires.Stat_Visualisation import palette_defaut, creation_graphique, creation_tableau, palette_daltonien
from utilitaires.Stat_Selection_Parametre import selection_parametre, selection_plage_date, selection_titre, selection_variable_filtre, selection_donnees_format_export
from utilitaires.Stat_Export_Resultat import gerer_export
from streamlit_extras.add_vertical_space import add_vertical_space
from utilitaires.Stat_mise_page import mise_forme_checkbox_radio, reduction_espace_titre_texte
import plotly.express as px
import tempfile


display_menu()

def show():
    """
    Affiche l'interface des statistiques.

    Cette fonction permet aux utilisateurs de sélectionner des paramètres, de visualiser des graphiques et des tableaux,
    et d'exporter les résultats sous forme de fichier ZIP. Elle inclut les étapes de chargement des données, de sélection
    des paramètres, et de génération des visualisations et des fichiers d'export.

    Parameters:
    None

    Returns:
    None
    """
        
    st.title("Statistiques")

    check_dependencies("Statistiques")

    # Chargement du fichier json resultats
    data=charger_fichier_json()
    # Extrait les données d'entrée (données observées) 
    df_donnees_entrees, entree, liste_unite, id_modele_entree=extraire_donnees_entree(data)
    # Extrait les données de prédiction et les données de KPI
    df_prediction, donnees_kpi, prediction=extraire_donnees_prediction_et_kpi(data, entree)
    # Fusione les données d'entréé et de prédiction pour créer un unique df
    df_final=fusionner_et_convertir(df_donnees_entrees, df_prediction)
    # Créé des listes et données nécessaires dans les filtres, affichage des boutons...
    (col_temps,liste_modeles_id,nb_modele,id_modele_moyen,
     unite_mesure_defaut,var_id,var_val,min_date,max_date,min_date_entree,max_date_entree,
            min_date_prediction,max_date_prediction)=obtenir_info_metadata(df_final, df_donnees_entrees, df_prediction, prediction,entree)
    
    # pour réduire espace entre les titres stMarkdown et les textes explicatifs
    reduction_espace_titre_texte()
   


   
    st.markdown("#### 🔧 Sélection des Paramètres à afficher") 
    st.markdown("<p style='font-size:14px; font-style:italic;'>Le graphique et le tableau des métriques se mettront à jour dynamiquement en fonction de vos sélections.</p>", unsafe_allow_html=True)
    
    # Initialisation des paramétres
    st.session_state.setdefault("affichage_modele_entree", True)
    st.session_state.setdefault("choix_temps", "Temps horaire")
    st.session_state.setdefault("affichage_ensemble_prediction", True)
    st.session_state.setdefault("affichage_moyenne_prediction", False)
    st.session_state.setdefault("selection_date", (min_date, max_date))
    st.session_state.setdefault("choix_unite", unite_mesure_defaut)
    #st.session_state.setdefault("export_donnees",True)

    selection_date = st.session_state["selection_date"]
    
    mise_forme_checkbox_radio()

    # Selection des parametres et variables
    affichage_modele_entree, choix_temps, affichage_ensemble_prediction, affichage_moyenne_prediction, choix_unite= selection_parametre(liste_unite,nb_modele,min_date,max_date)
    
    #création d'une liste pour simplifier la gestion de la selection des affichages
    selection_options = {"affichage_modele_entree": st.session_state.affichage_modele_entree,
                         "affichage_ensemble_prediction": st.session_state.affichage_ensemble_prediction,
                         "affichage_moyenne_prediction": st.session_state.affichage_moyenne_prediction}
    
    #st.session_state["selection_date"] = {"debut": debut_date, "fin": fin_date}

    
   
    #Filtre sur l'axe des temps
    st.markdown("📅 Fenêtre temporelle:")

    # Mise en forme et message rappelant les bornes temporelles 
    st.markdown(
    f"""
    <div style="background-color:#f5f5f5; padding:10px; border-radius:5px;">
        <em style="color:#333333; font-size:14px;">
        ⚠️ Attention :<br>
        Les données d’entrée sont disponibles entre <strong>{min_date_entree} et {max_date_entree}</strong>,
        et les prévisions entre <strong>{min_date_prediction} et {max_date_prediction}</strong>.<br>
        Si vous sélectionnez une période excluant certains intervalles,les données concernées ne seront pas affichées même si elles sont sélectionnées.
        </em>
    </div>
    """,
    unsafe_allow_html=True
    )

    # CHANGEMENT COULEUR SLIDER - NE MARCHE PAS A FINALISER
    couleur_slider = "#00FF00" 
        #if not mode_daltonien else "#FFD700"

    # pour changer couleur slider - ne marche pas
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

    #Slider pour sélectionner les plages horaires des données
    debut_date, fin_date = selection_plage_date(min_date, max_date)

    # pour éviter desynchronisation de session_state par rapport aux données du slider (normalement plus necessaire mais conservé)
    if (debut_date != st.session_state["selection_date"][0]) or (fin_date != st.session_state["selection_date"][1]):
        st.session_state["selection_date"] = (debut_date, fin_date)

     
    
    # Données et df filtrés suite à la sélection des parametres
    df_final_selection, df_kpi_selection, liste_donnees_filtre=selection_variable_filtre(id_modele_entree, selection_options, selection_date, df_final, donnees_kpi, liste_modeles_id, var_id, id_modele_moyen, choix_unite)


    # Création titre et label des axes suite à la sélection des parametres
    titre_graphe, label_x, label_y=selection_titre(selection_options, choix_temps, liste_unite, choix_unite)
    
    add_vertical_space(2) # ajout espace
    st.markdown("#### 📈 Affichage des Résultats")

    # Mise en page en deux colonnes
    col1, col2 = st.columns([2, 5])  # Col1 = 1/4 de la page, Col2 = 3/4
    with col1:

        st.markdown("#####  Rappel des Métriques")
        tab=creation_tableau (df_kpi_selection)
        #st.plotly_chart(tab, use_container_width=True)
        st.plotly_chart(tab)
        #st.markdown("<div style='margin-top: -50px;'></div>", unsafe_allow_html=True)

        
    with col2:
        st.markdown(f" ##### {titre_graphe}")
        fig=creation_graphique(df_final_selection, palette_daltonien, liste_donnees_filtre, var_id,choix_temps,var_val, label_x,label_y)
        st.plotly_chart(fig)
        #st.pyplot(fig)
        #st.markdown("<div style='margin-top: -50px;'></div>", unsafe_allow_html=True)
        

    #add_vertical_space(2)

    # Export des données selectionnées
    st.markdown("#### Sélection des paramètres d'export")
    st.markdown("<p style='font-size:14px; font-style:italic;'>Sélectionner les données et format à exporter puis cliquer sur le bouton pour les exporter en fichier zip.</p>", unsafe_allow_html=True)
    st.markdown(" ")
    


    export_options=selection_donnees_format_export()
    

    # Initialisation des états zip
    if "zip_ready" not in st.session_state:
        st.session_state.zip_ready = False
    if "zip_path" not in st.session_state:
        st.session_state.zip_path = None  # Chemin du fichier ZIP temporaire

    # controle pour vérifier si au moins un type de données et un format sont sélectionnés (cochés)
    export_possible = (any(export_options["donnees"].values()) and any(export_options["formats"].values()))

    # Bouton pour générer le fichier ZIP - grisé si export_possible ==False
    if st.button("📝 Générer le fichier ZIP", disabled= not export_possible): 
        gerer_export(df_final_selection, df_kpi_selection, export_options, fig, tab, titre_graphe)

                
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
