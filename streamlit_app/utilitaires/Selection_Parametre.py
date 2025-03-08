import streamlit as st
import pandas as pd


# Fonction pour afficher les propositions de sélection des parametres utilisés pour l'affichage du graphe et tableau
def selection_parametre(liste_unite,nb_modele):
    #st.markdown("#### 🔧 Sélection des Paramètres à afficher")

    # Initialisation des valeurs dans session_state
    st.session_state.setdefault("affichage_modele_entree", True)
    st.session_state.setdefault("choix_temps", "temps horaire")
    st.session_state.setdefault("affichage_ensemble_prediction", True)
    st.session_state.setdefault("affichage_moyenne_prediction", False)
    st.session_state.setdefault("choix_unite", liste_unite[0])
    

    # Ajout du CSS pour aligner checkboxes et boutons radio
    st.markdown(
    """
    <style>
    /* Alignement des checkboxes */
    div[data-testid="stCheckbox"] {
        margin-top: -_px;  /* Remonte toutes les checkboxes */
        margin-bottom: -5px; /* Réduit l'espace entre elles */
        display: flex;  /* Force l'alignement en ligne */
        align-items: center; /* Assure l'alignement vertical */
    }

    /* Corrige les marges spécifiques à la troisième checkbox */
    div[data-testid="stCheckbox"]:last-of-type {
        margin-top: -10px !important;  /* Remonte  la troisième checkbox */
    }

    /* Réduit  l'espace entre les boutons radio */
    div[data-testid="stRadio"] {
        margin-top: -38px;  /* Remonte légèrement les boutons radio */
        margin-bottom: -10px; /* Réduit l’espace vertical entre eux */
    }
   
    /* Vérifie que les contenants n'ajoutent pas de marges supplémentaires */
    div[data-testid="stMarkdownContainer"] {
        margin-bottom: -10px;
    }
    </style>
    """,
    unsafe_allow_html=True
    )


    #Decoupage de la page en colonnes pour mettre les propositions
    col1, col2, col3, col4 = st.columns([2,1,1,1])  
    
    with col1:
        st.markdown("📊Les données:")
        # Affichage de l'historique
        affichage_modele_entree=st.checkbox("Données d'entrée", value=True)
         # Affichage pour l'ensemble des prédictions
        affichage_ensemble_prediction = st.checkbox("Ensemble des prédictions", 
                                    value=st.session_state.affichage_ensemble_prediction)
        # Affichage de la moyenne si plusieurs prédictions
        affichage_moyenne_prediction = (st.checkbox("Moyenne des prédictions", 
                                    value=st.session_state.affichage_moyenne_prediction) 
                                    if nb_modele > 1 else False)
    

    with col2:
        # Axe de temps
        st.markdown("📅 L'axe temporel:")
        choix_temps = st.radio("", ["Temps horaire", "Temps relatif"], 
                               index=0 if st.session_state.choix_temps == "temps horaire" else 1).strip().lower()
        
    with col3:
        # Sélection de l'unité de mesure
        st.markdown("📅 L'unité:")
        choix_unite = st.radio("", liste_unite, index=liste_unite.index(st.session_state.choix_unite))
    
    
     # Mise à jour de session_state après sélection
    st.session_state.affichage_modele_entree = affichage_modele_entree
    st.session_state.choix_temps = choix_temps
    st.session_state.affichage_ensemble_prediction = affichage_ensemble_prediction
    st.session_state.affichage_moyenne_prediction = affichage_moyenne_prediction
    st.session_state.choix_unite = choix_unite

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
    
    return titre_graphe, label_x, label_y, df_kpi_selection, df_entrees_prevision_selection, choix_temps, liste_donnees_filtre


# fonction pour sélectionner les données d'export
def selection_donnee_export():
    st.write("Les données :")
    donnees_prevision="Données des prévisions"
    donnees_kpi="Métriques"

    # Initialisation des valeurs dans session_state
    st.session_state.setdefault("export_donnees", True)
    st.session_state.setdefault("export_kpi", True)
    st.session_state.setdefault("choix_donnees_export", [donnees_prevision, donnees_kpi])


    export_donnees = st.checkbox(donnees_prevision, value=st.session_state.export_donnees)
    export_kpi = st.checkbox(donnees_kpi, value=st.session_state.export_kpi)
    
    # Stocker les choix dans une liste
    choix_donnees_export = []
    if export_donnees:
        choix_donnees_export.append(donnees_prevision)
    if export_kpi:
        choix_donnees_export.append(donnees_kpi)
    
    # Mise à jour de session_state après sélection
    st.session_state.export_donnees = export_donnees
    st.session_state.export_kpi = export_kpi
    st.session_state.choix_donnees_export = choix_donnees_export

    return choix_donnees_export, donnees_prevision, donnees_kpi

# fonction pour sélectionner le format d'export
def selection_format_export():
    st.write("Les formats :")

    # Initialisation des valeurs dans session_state
    st.session_state.setdefault("export_format_csv", True)
    st.session_state.setdefault("export_format_pdf", True)
    st.session_state.setdefault("choix_format_export", ["CSV", "PDF"])


    export_format_csv = st.checkbox("CSV", value=st.session_state.export_format_csv)
    export_format_pdf = st.checkbox("PDF", value=st.session_state.export_format_pdf)
    
    # Stocker les choix dans une liste
    choix_format_export = []
    
    if export_format_csv:
        choix_format_export.append("CSV")
    
    if export_format_pdf:
        choix_format_export.append("PDF")

    # Mise à jour de session_state après sélection
    st.session_state.export_format_csv = export_format_csv
    st.session_state.export_format_pdf = export_format_pdf
    st.session_state.choix_format_export = choix_format_export

    return choix_format_export


