import streamlit as st
st.set_page_config(layout="wide")  # Active le mode large

from datetime import timedelta
from menu import display_menu

from dependency_manager import check_dependencies
from utilitaires.stat_chargement import charger_fichier_json, extraire_donnees_entree, extraire_donnees_prediction_et_kpi, fusionner_et_convertir, obtenir_info_metadata
from utilitaires.stat_visualisation import  creation_graphique, creation_tableau, palette_daltonien
from utilitaires.stat_selection_parametre import selection_parametre, selection_plage_date, selection_titre, selection_min_max_choix_temps, selection_variable_filtre, selection_donnees_format_export
from utilitaires.stat_export_resultat import gerer_export
from streamlit_extras.add_vertical_space import add_vertical_space
from utilitaires.mise_page import mise_forme_checkbox_radio, reduction_espace_titre_texte, afficher_bandeau_titre, style_SliderThumbValue
import plotly.express as px
import tempfile
import pandas as pd


display_menu()

def show():
    """
    Affiche la page Statistiques de la plateforme de prédiction de trafic réseau.

    Cette fonction :
    - Charge les données du fichier JSON (données d’entrée, prédictions, KPI),
    - Extrait et fusionne les données nécessaires à l’affichage,
    - Met en forme les paramètres sélectionnables (type de temps, unité, modèles à afficher),
    - Met à jour dynamiquement le graphique et le tableau des métriques en fonction des choix utilisateur,
    - Propose un système d’export multi-format (CSV, PNG, PDF) regroupé dans un fichier ZIP.

    Paramètres :
        Aucun

    Résultat retourné :
        Aucun
    """
    afficher_bandeau_titre()
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
     unite_mesure_defaut,var_id,var_val)=obtenir_info_metadata(df_final, df_donnees_entrees, df_prediction, prediction,entree)
    
    # pour réduire espace entre les titres stMarkdown et les textes explicatifs
    reduction_espace_titre_texte()
   

    st.markdown("#### 🔧 Sélection des Paramètres à afficher") 
    st.markdown("<p style='font-size:16px; font-style:italic;'>Le graphique et le tableau des métriques se mettront à jour dynamiquement en fonction de vos sélections.</p>", unsafe_allow_html=True)
    
    # Initialisation des paramétres session_state
    st.session_state.setdefault("affichage_modele_entree", True)
    st.session_state.setdefault("choix_temps", "Temps horaire")
    st.session_state.setdefault("affichage_ensemble_prediction", True)
    st.session_state.setdefault("affichage_moyenne_prediction", False)
    st.session_state.setdefault("choix_unite", unite_mesure_defaut)

    
    mise_forme_checkbox_radio()

    # Selection des parametres et variables
    affichage_modele_entree, choix_temps, affichage_ensemble_prediction, affichage_moyenne_prediction, choix_unite= selection_parametre(liste_unite,nb_modele)
    
    #création d'un dictionnaire des options sélectionnées pour les modeles pour simplifier la gestion de la selection des affichages
    selection_options = {"affichage_modele_entree": st.session_state.affichage_modele_entree,
                         "affichage_ensemble_prediction": st.session_state.affichage_ensemble_prediction,
                         "affichage_moyenne_prediction": st.session_state.affichage_moyenne_prediction}
    

    min_date, max_date, min_date_entree, max_date_entree ,min_date_prediction ,max_date_prediction=selection_min_max_choix_temps(choix_temps, df_final,df_donnees_entrees,df_prediction)

    # Détermine les valeurs par défaut du slider en fonction du type de temps sélectionné 
    if choix_temps == "temps horaire":
        valeur_defaut = (df_final["temps horaire"].min().to_pydatetime(), df_final["temps horaire"].max().to_pydatetime())
    else:
        valeur_defaut = (int(df_final["temps relatif"].min()), int(df_final["temps relatif"].max()))

    # Si le type de temps sélectionné a changé (horaire ↔ relatif), 
    # on force la réinitialisation de la plage temporelle sélectionnée pour éviter des erreurs de type
    if "selection_date" in st.session_state:
        ancien_type = type(st.session_state["selection_date"][0])
        nouveau_type = type(valeur_defaut[0])
        if ancien_type != nouveau_type:
            st.session_state["selection_date"] = valeur_defaut
    else:
        st.session_state["selection_date"] = valeur_defaut
    
    selection_date = st.session_state["selection_date"]
    
    add_vertical_space(1)

    #Filtre sur l'axe des temps
    st.markdown("Fenêtre temporelle 📅 :")

    # Mise en forme et message rappelant les bornes temporelles 
    st.markdown(
    f"""
    <div style="background-color:#f5f5f5; padding:10px; border-radius:5px;border: 2px solid orange;">
        <em style="color:#333333; font-size:14px;">
        ⚠️ Attention :<br>
        En <strong>{choix_temps}</strong>, les données d’entrée sont disponibles entre <strong>{min_date_entree} et {max_date_entree}</strong>,
        et les prévisions entre <strong>{min_date_prediction} et {max_date_prediction}</strong>.<br>
        Si vous sélectionnez une période excluant certains intervalles,les données concernées ne seront pas affichées même si elles sont sélectionnées.
        </em>
    </div>
    """,
    unsafe_allow_html=True
    )

    style_SliderThumbValue()

   
    #Slider pour sélectionner les plages horaires des données
    debut_date, fin_date = selection_plage_date(min_date, max_date, choix_temps)
    
    
    # Données et df filtrés suite à la sélection des parametres
    df_final_selection, df_kpi_selection, liste_donnees_filtre=selection_variable_filtre(id_modele_entree, 
                    selection_options, selection_date, df_final, donnees_kpi, liste_modeles_id, 
                    var_id, id_modele_moyen, choix_unite, choix_temps)


    # Création titre et label des axes suite à la sélection des parametres
    titre_graphe, label_x, label_y=selection_titre(selection_options, choix_temps, liste_unite, choix_unite)
    
    # Ajouter une séparation
    st.markdown("---")

    add_vertical_space(1) # ajout espace

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
        st.markdown(f" ##### Graphique de prédiction des différents modèles obtenus")
        fig=creation_graphique(df_final_selection, palette_daltonien, liste_donnees_filtre, var_id,choix_temps,var_val, label_x,label_y)
        st.plotly_chart(fig)
        #st.pyplot(fig)
        #st.markdown("<div style='margin-top: -50px;'></div>", unsafe_allow_html=True)
        
    #ajoute un espace
    add_vertical_space(2)

    # Ajouter une séparation
    st.markdown("---")

    # Export des données selectionnées
    st.markdown("#### 📦 Sélection des paramètres d'export")
    st.markdown("<p style='font-size:16px; font-style:italic;'>Sélectionner les données et formats à exporter puis cliquer sur le bouton pour les exporter en fichier zip.</p>", unsafe_allow_html=True)
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

    # Si modification dans les sélections, on invalide l'ancien zip (zip_ready=False) pour forcer sa régénération
    if any(st.session_state[key] for key in {**export_options["formats"], **export_options["donnees"]}):
        st.session_state.zip_ready = False  # Force la régénération
        st.session_state.zip_path = None  # Supprime le chemin du fichier
  

    st.session_state.valid_statistiques = True

if __name__ == "__main__":
    show()
