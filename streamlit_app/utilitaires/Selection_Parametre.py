import streamlit as st
#from streamlit_datetime_range_picker import datetime_range_picker
from datetime import timedelta
import pandas as pd



def create_checkbox_group(options):
    """
    Genere groupe de checkbox basé sur une liste (options) 
    avec comme clé le nom correspondant dans `st.session_state`
    """
    return {key: st.checkbox(label, key=key) for key, label in options.items()}

# Fonction pour afficher les propositions de sélection des parametres utilisés pour l'affichage du graphe et tableau
def selection_parametre(liste_unite,nb_modele,min_date,max_date, selection_date):
    """ Gére l'affichage par checkbox, bouton radio des differentes options necessaires 
    pour la sélection des parametres utiles pour le graphe et tableau,
    """

    #Decoupage de la page en colonnes pour mettre les propositions
    col1, col2, col3, col4 = st.columns([2,1,1,1])  
    
    with col1:
        st.markdown("📊Les données:")
        # Affichage des checkbox pour déterminer les données
        checkboxes = create_checkbox_group({
            "affichage_modele_entree": "Données d'entrée",
            "affichage_ensemble_prediction": "Ensemble des prédictions"
        })
        # Affichage de la moyenne des prédictions uniquement si plusieurs modèles existent
        affichage_moyenne_prediction = (st.checkbox("Moyenne des prédictions", key="affichage_moyenne_prediction") 
                                        if nb_modele > 1 else False)
        
    with col2:
        # Axe de temps
        st.markdown("📅 Type de temps:")
        #choix_temps = create_radio_group(["Temps horaire", "Temps relatif"], "choix_temps")
        choix_temps = st.radio("",["Temps horaire", "Temps relatif"],key="choix_temps")
        choix_temps = choix_temps.strip().lower()

    with col3:
        # Sélection de l'unité de mesure
        st.markdown("📅 Unité de mesure:")
        choix_unite = st.radio("", liste_unite, key="choix_unite")
    
    return checkboxes["affichage_modele_entree"], choix_temps, checkboxes["affichage_ensemble_prediction"], affichage_moyenne_prediction, choix_unite

def selection_plage_date(min_date, max_date):
    # Slider pour sélectionner la plage temporelle
    debut_date, fin_date = st.slider(
        "",
        min_value=min_date,
        max_value=max_date,
        #value=st.session_state["selection_date"],  # pas necessaire, doublon avec key
        step=timedelta(seconds=1),
        format="YYYY-MM-DD HH:mm:ss",
        key="selection_date"  # synchronisation automatique
    )

    return debut_date, fin_date



def selection_variable_filtre(id_modele_entree, selection_options, selection_date, df_entrees_prevision, donnees_kpi, liste_modeles_id, var_id, id_modele_moyen,):
    """
    Sélectionne les variables et données filtrées en fonction des paramétres choisis par l'utilisateur.
    Retourne les DataFrames filtrés pour le graphique et le tableau.
    """

    # Déduction des valeurs en fonction des options sélectionnées
    modele_moyen = id_modele_moyen if selection_options["affichage_moyenne_prediction"] else []
    liste_modele_ensemble = liste_modeles_id if selection_options["affichage_ensemble_prediction"] else []
    liste_modele_entree = id_modele_entree if selection_options["affichage_modele_entree"] else []

    # Liste complète des modèles à afficher
    liste_modele = modele_moyen + liste_modele_ensemble
    liste_donnees_filtre = liste_modele_entree + liste_modele

    # Selection plage temps

    # Filtrage des données
    df_entrees_prevision_selection = df_entrees_prevision[(df_entrees_prevision[var_id].isin(liste_donnees_filtre)) &
                                                          (df_entrees_prevision['temps horaire'] >= pd.Timestamp(selection_date[0])) &
                                                          (df_entrees_prevision['temps horaire'] <= pd.Timestamp(selection_date[1]))]
   
    # verification si liste modele change apres le filtrage
    liste_modele_filtre_selection = df_entrees_prevision_selection['id donnee'].unique().tolist()
    if set(liste_modele_filtre_selection) !=set(liste_donnees_filtre):
        if set(liste_modele_filtre_selection)==set(liste_modele):
            st.toast("Les données d'entrée ont été exclues par votre sélection temporelle.", icon="⚠️")
        elif not liste_modele_filtre_selection:  # liste complètement vide
            st.toast("Toutes les données ont été exclues par votre sélection temporelle.", icon="⚠️")
        else:
            st.toast("Les données de prédiction ont été exclues par votre sélection temporelle.", icon="⚠️")
            liste_modele=[] 
    liste_donnees_filtre=liste_modele_filtre_selection

    

    
    df_kpi = pd.DataFrame(donnees_kpi)
    if liste_modele:
        df_kpi_selection = df_kpi[df_kpi[var_id].isin(liste_modele)]
    else:
        df_kpi_selection = pd.DataFrame(columns=df_kpi.columns)


    return df_entrees_prevision_selection, df_kpi_selection, liste_donnees_filtre

def selection_titre(selection_options, choix_temps, mesure_format, choix_unite):
    """
    Génère le titre du graphique et les labels des axes en fonction des options sélectionnées.
    """

    # Construction du titre du graphique
    titre_parts = []
    if selection_options["affichage_modele_entree"]:
        titre_parts.append("Données d'entrée")
    if selection_options["affichage_ensemble_prediction"]:
        titre_parts.append("Ensemble des Prédictions")
    if selection_options["affichage_moyenne_prediction"]:
        titre_parts.append("Moyenne des Prédictions")

    titre_graphe = "Graphique des " + ", ".join(titre_parts)

    # format déduit de l'unité choisi
    index_unite= mesure_format['unite mesure'].index(choix_unite)
    choix_format = "" if not mesure_format["format mesure"] else mesure_format["format mesure"][index_unite]

    # Labels des axes
    label_x = choix_temps
    label_y = f"{choix_format} {choix_unite}"

    return titre_graphe, label_x, label_y

def selection_donnees_format_export():
    """
    Génère les options de sélection pour l'exportation des données et des formats.
    """

    col1, col2, col3 = st.columns([1,1,2])  

    with col1:
        st.write("📊 Les données :")

        # Checkbox pour les données à exporter
        donnees_options = create_checkbox_group({
            "export_prevision": "Données des prévisions",
            "export_kpi": "Métriques"
        })

    with col2:
        st.write("📂 Les formats :")

        # Checkbox pour les formats d'export
        format_options = create_checkbox_group({
            "export_format_csv": "CSV",
            "export_format_pdf": "PDF",
            "export_format_png": "PNG"
        })

    # 
    return {
        "donnees": donnees_options,
        "formats": format_options
    }



