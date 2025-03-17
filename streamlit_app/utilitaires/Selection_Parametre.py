import streamlit as st
import pandas as pd


# Fonction pour afficher les propositions de sélection des parametres utilisés pour l'affichage du graphe et tableau
def selection_parametre(liste_unite,nb_modele):
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






    #st.markdown('<div class="main-container">', unsafe_allow_html=True)

    #Decoupage de la page en colonnes pour mettre les propositions
    col1, col2, col3, col4 = st.columns([2,1,1,1])  
    
    with col1:
        st.markdown("📊Les données:")
        # Affichage de l'historique
        #affichage_modele_entree=st.checkbox("Données d'entrée", value=True)
        affichage_modele_entree = st.checkbox("Données d'entrée", key='affichage_modele_entree')
         # Affichage pour l'ensemble des prédictions
        #affichage_ensemble_prediction = st.checkbox("Ensemble des prédictions", 
        #                            value=st.session_state.affichage_ensemble_prediction)
        affichage_ensemble_prediction = st.checkbox("Ensemble des prédictions", key='affichage_ensemble_prediction')
        # Affichage de la moyenne si plusieurs prédictions
        affichage_moyenne_prediction = (st.checkbox("Moyenne des prédictions", key="affichage_moyenne_prediction") 
                                        if nb_modele > 1 else False)
        
        

    with col2:
        # Axe de temps
        st.markdown("📅 L'axe temporel:")
       
        choix_temps = st.radio("",["Temps horaire", "Temps relatif"],key="choix_temps")
        choix_temps = choix_temps.strip().lower()

    with col3:
        # Sélection de l'unité de mesure
        st.markdown("📅 L'unité:")
        choix_unite = st.radio("", liste_unite, key="choix_unite")

    #st.markdown('</div>', unsafe_allow_html=True)
    
    return affichage_modele_entree, choix_temps, affichage_ensemble_prediction, affichage_moyenne_prediction, choix_unite


# fonction pour identifier les variables et filtre suite à la sélection
def selection_variable(id_modele_entree, choix_temps, affichage_modele_entree, affichage_moyenne_prediction, affichage_ensemble_prediction, df_entrees_prevision, donnees_kpi, liste_modeles_id,mesure_format,choix_unite, var_id,id_modele_moyen):
    
    # déduit de la selection : ensemble des données liées aux Prédictions et/ou moyenne uniquement
    modele_moyen,titre_part_moyen=(id_modele_moyen,"Moyenne des Prédictions") if affichage_moyenne_prediction else ([],"")
    liste_modele_ensemble,titre_part_ensemble=(liste_modeles_id,"Ensemble des Prédictions") if affichage_ensemble_prediction else ([],"")
     
    #déduit de l'affichage historique:
    liste_modele_entree,titre_part_modele_entree=(id_modele_entree,"Données d'entrée") if affichage_modele_entree else ([],"")

    # format déduit de l'unité choisi
    index_mesure= mesure_format['unite mesure'].index(choix_unite)
    choix_format = "" if not mesure_format["format mesure"] else mesure_format["format mesure"][index_mesure]


    # titre du graphique construit à partir des différentes sélection
    titre_graphe = "Graphique des " + ", ".join(filter(None, [titre_part_modele_entree, titre_part_ensemble, titre_part_moyen]))

    # affichage des labels
    label_x = choix_temps # selon sélection de l'xe temporelle
    label_y = f"{choix_format} {choix_unite} " #selon sélection de l'unité de mesure et déduction du format associé
    
    #selection des données pour le graphique
    liste_modele=modele_moyen+liste_modele_ensemble
    liste_donnees_filtre=liste_modele_entree+liste_modele
    df_entrees_prevision_selection=df_entrees_prevision[df_entrees_prevision[var_id].isin(liste_donnees_filtre)]

    #Selection des données pour le tableau des Indicateurs
    df_kpi = pd.DataFrame(donnees_kpi) # transformation en dataframe
    df_kpi_selection=df_kpi[df_kpi[var_id].isin(liste_modele)]
    
    return titre_graphe, label_x, label_y, df_kpi_selection, df_entrees_prevision_selection, liste_donnees_filtre




# fonction pour sélectionner les données d'export

def selection_donnees_format_export():

    col1, col2, col3 = st.columns([1,1,2])  
    with col1:
        st.write("Les données :")
        donnees_prevision="Données des prévisions"
        donnees_kpi="Métriques"
        export_donnees = st.checkbox(donnees_prevision, key='export_donnees')
        export_kpi = st.checkbox(donnees_kpi, key='export_kpi')
    
        # Stocker les choix dans une liste
        choix_donnees_export = []
        if export_donnees:
            choix_donnees_export.append(donnees_prevision)
        if export_kpi:
            choix_donnees_export.append(donnees_kpi)
    

    with col2:
        # sélection du format d'export
        st.write("Les formats :")

        export_format_csv = st.checkbox("CSV", key='export_format_csv')
        export_format_pdf = st.checkbox("PDF", key='export_format_pdf')
        export_format_png = st.checkbox("PNG", key='export_format_png')
    
        # Stocker les choix dans une liste
        choix_format_export = []
    
        if export_format_csv:
            choix_format_export.append("CSV")
        
        if export_format_pdf:
            choix_format_export.append("PDF")
        
        if export_format_png:
            choix_format_export.append("PNG")
        
    #return choix_format_export, choix_donnees_export, donnees_prevision, donnees_kpi,dpi_value
    return choix_format_export, choix_donnees_export, donnees_prevision, donnees_kpi
