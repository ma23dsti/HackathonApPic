import streamlit as st
#from streamlit_datetime_range_picker import datetime_range_picker
from datetime import timedelta
import pandas as pd



def create_checkbox_group(options):
    """
    Génère un groupe de checkbox basé sur une liste (options) 
    avec comme clé le nom correspondant dans st.session_state
    """
    return {key: st.checkbox(label, key=key) for key, label in options.items()}



def selection_parametre(liste_unite, nb_modele):
    """
    Gère l'affichage des paramètres sélectionnables via checkbox et radio bouton,
    nécessaires pour l'affichage du graphe et du tableau.

    Args:
        liste_unite (list): Liste des unités à afficher dans le bouton radio.
        nb_modele (int): Nombre de modèles disponibles (utile pour afficher la moyenne).
        min_date, max_date: (plus utilisés  - à sup)

    Returns:
        Tuple contenant les états sélectionnés :
        - affichage_modele_entree (bool)
        - choix_temps (str, formaté en minuscule)
        - affichage_ensemble_prediction (bool)
        - affichage_moyenne_prediction (bool ou False)
        - choix_unite (str)
    """

    # Mise en page avec 4 colonnes (ratios personnalisés)
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    # Colonne 1 : Affichage des données
    with col1:
        st.markdown("Les données 📊:")

        # Affichage des cases à cocher pour les types de données
        checkboxes = create_checkbox_group({
            "affichage_modele_entree": "Données d'entrée",
            "affichage_ensemble_prediction": "Ensemble des prédictions"
        })

        # Affichage conditionnel de la moyenne des prédictions
        affichage_moyenne_prediction = (
            st.checkbox("Moyenne des prédictions", key="affichage_moyenne_prediction") 
            if nb_modele > 1 else False
        )

    # Colonne 2 : Choix du type de temps
    with col2:
        st.markdown("Type de temps 📅:")
        choix_temps = st.radio("", ["Temps horaire", "Temps relatif"], key="choix_temps")
        choix_temps = choix_temps.strip().lower()  # normalisation du choix

    # Colonne 3 : Choix de l’unité de mesure
    with col3:
        st.markdown("Unité de mesure 📏:")
        choix_unite = st.radio("", liste_unite, key="choix_unite")

    # Retour des choix utilisateur
    return (
        checkboxes["affichage_modele_entree"],
        choix_temps,
        checkboxes["affichage_ensemble_prediction"],
        affichage_moyenne_prediction,
        choix_unite
    )

def selection_min_max_choix_temps(choix_temps, df_final,df_donnees_entrees,df_prediction):
    if choix_temps == "temps horaire":
        min_date = df_final[choix_temps].min().to_pydatetime()
        max_date = df_final[choix_temps].max().to_pydatetime()
        min_date_entree = df_donnees_entrees[choix_temps].min().to_pydatetime()  # date min des entrées
        max_date_entree = df_donnees_entrees[choix_temps].max().to_pydatetime()  # date max des entrées
        min_date_prediction = df_prediction[choix_temps].min()  # date min prédictions
        max_date_prediction = df_prediction[choix_temps].max()  # date max prédictions
    else:
        min_date = df_final[choix_temps].min()
        max_date = df_final[choix_temps].max()
        min_date_entree = df_donnees_entrees[choix_temps].min()
        max_date_entree = df_donnees_entrees[choix_temps].max()
        min_date_prediction = df_prediction[choix_temps].min() 
        max_date_prediction = df_prediction[choix_temps].max()
    
    return min_date, max_date, min_date_entree, max_date_entree ,min_date_prediction ,max_date_prediction

    
def selection_plage_date(min_date, max_date, choix_temps):
    """ Genere un slider pour sélectionner la plage temporelle min et max en fonction de la selection du type de temps"""
    
    # paramètres du slider selon choix_temps
    if choix_temps == "temps horaire":
        step = timedelta(seconds=1)
        slider_format = "YYYY-MM-DD HH:mm:ss"
    else:
        step = 1
        slider_format = None
    debut_date, fin_date = st.slider(
        "",
        min_value=min_date,
        max_value=max_date,
        #step=timedelta(seconds=1), # figée à 1 car c'est la périodicité que nous avons pris - pour la rendre dynamique cf fichier resultats.json periodicite_mesure
        #format="YYYY-MM-DD HH:mm:ss",
        format=slider_format if slider_format else None,
        step=step,
        key="selection_date"  # synchronisation automatique
    )
   

    return debut_date, fin_date


def generer_listes_modeles(selection_options, id_modele_entree, id_modele_moyen, liste_modeles_id):
    """
    Déduction des valeurs en fonction des options sélectionnées.
    Retourne les trois listes : moyenne, ensemble, entrée, ainsi que la liste combinée.
    """
    modele_moyen = id_modele_moyen if selection_options["affichage_moyenne_prediction"] else []
    liste_modele_ensemble = liste_modeles_id if selection_options["affichage_ensemble_prediction"] else []
    liste_modele_entree = id_modele_entree if selection_options["affichage_modele_entree"] else []

    liste_donnees_filtre = liste_modele_entree + modele_moyen + liste_modele_ensemble
    return liste_donnees_filtre, modele_moyen, liste_modele_ensemble


def filtrer_df_final(df_final, var_id, choix_unite, choix_temps, selection_date, liste_donnees_filtre):
    """
    Filtrage du DataFrame principal selon les paramètres sélectionnés.
    """
    #st.write("choix_temps :", choix_temps)
    #st.write(" type de selection_date[0] :", type(selection_date[0]))
    #st.write(" valeur selection_date[0] :", selection_date[0])

    # identification des bornes_temps inf et max
    if choix_temps == "temps horaire":
        borne_temps_inf = selection_date[0]
        borne_temps_sup = selection_date[1]
    else:
        borne_temps_inf = int(selection_date[0])
        borne_temps_sup = int(selection_date[1])

    # Filtrage des données
    df_final_selection = df_final[
        (df_final[var_id].isin(liste_donnees_filtre)) &
        (df_final['unite mesure'] == choix_unite) &
        (df_final[choix_temps] >= borne_temps_inf) &
        (df_final[choix_temps] <= borne_temps_sup)
    ]
    
    return df_final_selection

def verifier_modele_exclus(df_final_selection, liste_donnees_filtre, modele_moyen, liste_modele_ensemble):
    """
    Vérifie si la liste des modèles est impactée par la sélection temporelle et message d'alerte selon le cas.
    Met à jour les listes en conséquence.
    """
    liste_modele_filtre_selection = df_final_selection['id donnee'].unique().tolist()
    if set(liste_modele_filtre_selection) != set(liste_donnees_filtre):
        if set(liste_modele_filtre_selection) == set(modele_moyen + liste_modele_ensemble):
            st.toast("Les données d'entrée ont été exclues par votre sélection temporelle.", icon="⚠️")
        elif not liste_modele_filtre_selection:  # liste complètement vide
            st.toast("Toutes les données ont été exclues par votre sélection temporelle.", icon="⚠️")
        else:
            st.toast("Les données de prédiction ont été exclues par votre sélection temporelle.", icon="⚠️")
            modele_moyen = []
            liste_modele_ensemble = []
    
    return liste_modele_filtre_selection, modele_moyen, liste_modele_ensemble

def selection_variable_filtre(id_modele_entree, selection_options, selection_date, df_final, donnees_kpi,
                               liste_modeles_id, var_id, id_modele_moyen, choix_unite, choix_temps):
    """
    Sélectionne les variables et données filtrées en fonction des paramètres choisis par l'utilisateur.
    Retourne les DataFrames filtrés pour le graphique et le tableau.
    """

    # Étape 1 : déterminer les modèles à afficher
    liste_donnees_filtre, modele_moyen, liste_modele_ensemble = generer_listes_modeles(
        selection_options, id_modele_entree, id_modele_moyen, liste_modeles_id
    )

    # Étape 2 : filtrer les données
    df_final_selection = filtrer_df_final(
        df_final, var_id, choix_unite, choix_temps, selection_date, liste_donnees_filtre
    )

    # Étape 3 : gérer les exclusions dues à la plage temporelle
    liste_donnees_filtre, modele_moyen, liste_modele_ensemble = verifier_modele_exclus(
        df_final_selection, liste_donnees_filtre, modele_moyen, liste_modele_ensemble
    )

    # Étape 4 : filtrage des données KPI
    df_kpi = pd.DataFrame(donnees_kpi)
    if modele_moyen + liste_modele_ensemble:
        df_kpi_selection = df_kpi[df_kpi[var_id].isin(modele_moyen + liste_modele_ensemble)]
    else:
        df_kpi_selection = pd.DataFrame(columns=df_kpi.columns)

    return df_final_selection, df_kpi_selection, liste_donnees_filtre



def selection_titre(selection_options, choix_temps, liste_unite, choix_unite):
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

    

    # Labels des axes
    label_x = choix_temps
    #label_y = f"{choix_format} {choix_unite}"
    label_y = f" {choix_unite}" # plotly affiche les axes en KMGT

    return titre_graphe, label_x, label_y

def selection_donnees_format_export():
    """
    Génère les options de sélection pour l'exportation des données et des formats.
    """

    col1, col2, col3 = st.columns([1,1,2])  # col1:1/4 ...

    with col1:
        st.write("Les données 📊:")

        # Checkbox pour les données à exporter
        donnees_options = create_checkbox_group({
            "export_prediction": "Données des prédictions",
            "export_kpi": "Métriques"
        })

    with col2:
        st.write("Les formats 📂:")

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



