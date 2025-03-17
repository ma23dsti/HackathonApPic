import streamlit as st
st.set_page_config(layout="wide")  # Active le mode large


from menu import display_menu
from dependency_manager import check_dependencies
from utilitaires.Chargement import chargement_donnees
from utilitaires.Visualisation_Stat import palette_defaut, creation_graphique, creation_tableau
from utilitaires.Selection_Parametre import selection_parametre, selection_variable, selection_donnees_format_export
from utilitaires.Export_Resultat import export_data_zip
from streamlit_extras.add_vertical_space import add_vertical_space


display_menu()

def show():
    st.title("Statistiques")

    check_dependencies("Statistiques")

    # en attendant d'avoir les vraies valeurs chargement de données pour faire des essais - Charger les données au démarrage
    df_entrees_prevision, donnees_kpi, col_temps, liste_modeles_id,nb_modele,id_modele_moyen,id_historique,liste_unite,mesure_format, var_id, var_val=chargement_donnees()
    
    
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
    st.session_state.setdefault("choix_unite", liste_unite[0])
    #st.session_state.setdefault("export_donnees",True)
    


    # Selection des parametres et variables
    affichage_historique, choix_temps, affichage_ensemble_prediction, affichage_moyenne_prediction, choix_unite= selection_parametre(liste_unite,nb_modele)
    titre_graphe, label_x, label_y, df_kpi_selection, df_entrees_prevision_selection, liste_donnees_filtre=selection_variable(id_historique, choix_temps, affichage_historique, affichage_moyenne_prediction, affichage_ensemble_prediction, df_entrees_prevision, donnees_kpi, liste_modeles_id,mesure_format,choix_unite, var_id,id_modele_moyen)

    add_vertical_space(2)
    st.markdown("#### 📈 Affichage des Résultats")

    # Mise en page en deux colonnes
    col1, col2 = st.columns([2, 5])  # Col1 = 1/4 de la page, Col2 = 3/4
    with col1:
        st.markdown("#####  Rappel des Métriques")
        tab=creation_tableau (df_kpi_selection)
        st.plotly_chart(tab)

        
    with col2:
        st.markdown(f" ##### {titre_graphe}")
        fig=creation_graphique(df_entrees_prevision_selection, palette_defaut, liste_donnees_filtre, var_id,choix_temps,var_val, label_x,label_y)
        st.pyplot(fig)
    
    add_vertical_space(2)

    #Export des données selectionnées

    st.markdown("#### Sélection des paramètres d'export")
    st.markdown("<p style='font-size:14px; font-style:italic;'>Sélectionner les données et format à exporter puis cliquer sur le bouton pour les exporter en fichier zip.</p>", unsafe_allow_html=True)
    st.markdown(" ")
    


    choix_format_export, choix_donnees_export, donnees_prevision, donnees_kpi=selection_donnees_format_export()
    # creation du fichier zip avec les éléments selectionnés
    zip_file = export_data_zip(df_entrees_prevision_selection, df_kpi_selection, choix_donnees_export, choix_format_export,donnees_prevision, donnees_kpi,fig, tab, titre_graphe)
    #zip_file = export_data_zip(df_entrees_prevision_selection, df_kpi_selection, choix_donnees_export, choix_format_export,donnees_prevision, donnees_kpi,fig, tab, titre_graphe,dpi_value)
    # Bouton d'export avec les éléments sélectionnés
    st.download_button(
        label="Télécharger le fichier ZIP",
        data=zip_file,
        file_name="export_resultats.zip",
        mime="application/zip"
    )
    st.session_state.valid_statistiques = True

if __name__ == "__main__":
    show()
