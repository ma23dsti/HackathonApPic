import streamlit as st
#from streamlit_datetime_range_picker import datetime_range_picker
from datetime import timedelta
import pandas as pd



def create_checkbox_group(options):
    """
    Génère dynamiquement un groupe de cases à cocher (checkbox) dans Streamlit à partir d’un dictionnaire d’options.

    Chaque clé du dictionnaire devient la clé d'état dans st.session_state,
    et chaque valeur devient l’étiquette visible dans l’interface utilisateur.

    Paramètre :
        options (dict) : Dictionnaire {clé: étiquette} représentant les options à afficher sous forme de cases à cocher.

    Résultat retourné :
        - dict : Dictionnaire {clé: booléen} indiquant pour chaque option si la case a été cochée ou non.
    """
    return {key: st.checkbox(label, key=key) for key, label in options.items()}



def selection_parametre(liste_unite, nb_modele):
    """
    Gère l'affichage des paramètres sélectionnables via checkbox et radio bouton,
    nécessaires pour l'affichage du graphe et du tableau.

    Paramètres :
        liste_unite (list[str]): Liste des unités à afficher dans le bouton radio.
        nb_modele (int): Nombre de modèles disponibles (utile pour l'affichage ou non de la moyenne des predictions).
    Remarque :
        La liste des types de temps n’est pas passée en paramètre car l’option est figée 
        directement dans l’appel au bouton radio.

    Retourne:
        Tuple contenant les états sélectionnés :
        - bool : Affichage des données d'entrée (coché ou non)
        - str : Type de temps sélectionné ('temps horaire' ou 'temps relatif', en minuscules)
        - bool : Affichage de l’ensemble des prédictions
        - bool : Affichage de la moyenne des prédictions (False si un seul modèle)
        - str : Unité de mesure sélectionnée
       
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
    """
    Détermine les dates ou valeur minimum et maximum à afficher dans le sélecteur de plage temporelle,
    en fonction du type de temps choisi par l'utilisateur.

    Paramètres :
        choix_temps (str) : Type de temps sélectionné ('temps horaire' ou 'temps relatif').
        df_final (pd.DataFrame) : Données combinées (entrées et prédictions).
        df_donnees_entrees (pd.DataFrame) : Données d'entrée uniquement.
        df_prediction (pd.DataFrame) : Données de prédiction uniquement.

    Résultats retournés :
        - datetime ou int : Date/valeur minimale du jeu de données global.
        - datetime ou int : Date/valeur maximale du jeu de données global.
        - datetime ou int : Date/valeur minimale des données d'entrée.
        - datetime ou int : Date/valeur maximale des données d'entrée.
        - datetime ou int : Date/valeur minimale des prédictions.
        - datetime ou int : Date/valeur maximale des prédictions.
    """
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
    """
    Affiche un curseur (slider) interactif permettant de sélectionner une plage temporelle
    en fonction des valeurs du type de temps choisi : temps horaire (datetime) ou temps relatif (entier).

    Paramètres :
        min_date (datetime ou int) : Valeur minimale à afficher dans le slider.
        max_date (datetime ou int) : Valeur maximale à afficher dans le slider.
        choix_temps (str) : Type de temps sélectionné par l'utilisateur ('temps horaire' ou 'temps relatif').

    Résultats retournés :
        - datetime ou int : Date/valeur de début sélectionné.
        - datetime ou int : Date/valeur de fin sélectionné.
    """
   
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
    Génère dynamiquement les listes de modèles à afficher selon les options sélectionnées par l'utilisateur.

    Cette fonction permet de déterminer :
    - si les données d'entrée doivent être affichées,
    - si la moyenne des prédictions est activée,
    - si l'ensemble des modèles est sélectionné.

    Paramètres :
        selection_options (dict) : Dictionnaire contenant les options d'affichage sélectionnées
                                   ( "affichage_modele_entree", "affichage_moyenne_prediction", etc.).
        id_modele_entree (list[str]) : Identifiant(s) des données d'entrée.
        id_modele_moyen (list[str]) : Identifiant du modèle moyen.
        liste_modeles_id (list[str]) : Liste des identifiants des modèles de prédiction.

    Résultats retournés :
        - list[str] : Liste combinée des identifiants à afficher (entrée, moyenne, ensemble).
        - list[str] : Liste contenant uniquement l’identifiant du modèle moyen (ou vide si non sélectionné).
        - list[str] : Liste des identifiants des modèles de prédiction (ou vide si non sélectionnés).
    """
    modele_moyen = id_modele_moyen if selection_options["affichage_moyenne_prediction"] else []
    liste_modele_ensemble = liste_modeles_id if selection_options["affichage_ensemble_prediction"] else []
    liste_modele_entree = id_modele_entree if selection_options["affichage_modele_entree"] else []

    liste_donnees_filtre = liste_modele_entree + modele_moyen + liste_modele_ensemble
    return liste_donnees_filtre, modele_moyen, liste_modele_ensemble


def filtrer_df_final(df_final, var_id, choix_unite, choix_temps, selection_date, liste_donnees_filtre):
    """
    Filtre le DataFrame principal (`df_final`) en fonction des paramètres sélectionnés par l'utilisateur :
    - les identifiants des modèles ou données d’entrée à afficher,
    - l’unité de mesure choisie,
    - la période temporelle sélectionnée.

    Paramètres :
        df_final (pd.DataFrame) : DataFrame global contenant les données d'entrée et de prédiction.
        var_id (str) : Nom de la colonne identifiant chaque série (modèle ou entrée).
        choix_unite (str) : Unité de mesure sélectionnée.
        choix_temps (str) : Type de temps sélectionné.
        selection_date (tuple) : Tuple contenant les bornes inférieure et supérieure du temps sélectionné.
        liste_donnees_filtre (list[str]) : Liste des identifiants des modeles sélectionnés

    Résultat retourné :
        - pd.DataFrame : Sous-ensemble filtré de `df_final` répondant aux critères sélectionnés.
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
    Vérifie si la sélection temporelle a entraîné l’exclusion de certaines données (entrée ou prédictions).

    Si une ou plusieurs séries ont été exclues (c’est-à-dire absentes du DataFrame filtré),
    un message d’alerte est affiché dans l’interface utilisateur avec un message toast.
    Cette fonction met également à jour les listes de modèles pour éviter tout affichage incohérent.

    Paramètres :
        df_final_selection (pd.DataFrame) : Données filtrées selon la plage temporelle.
        liste_donnees_filtre (list[str]) : Liste initiale des identifiants de modele devant être affichés.
        modele_moyen (list[str]) : Identifiant du modèle moyen, s'il était sélectionné.
        liste_modele_ensemble (list[str]) : Liste des identifiants des modèles de prédiction.

    Résultats retournés :
        - list[str] : Liste mise à jour des identifiants réellement présents dans la sélection.
        - list[str] : Liste actualisée du modèle moyen (vide si exclu).
        - list[str] : Liste actualisée des modèles de prédiction (vide si exclus).
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
    Filtre les données d’entrée, les prédictions et les KPI selon les paramètres sélectionnés 
    par l’utilisateur.

    Cette fonction applique une série d'étapes :
    1. Génère la liste des identifiants de données à afficher (entrée, moyenne, prédictions).
    2. Filtre les données globales `df_final` en fonction de ces identifiants, de l’unité et du temps.
    3. Vérifie si certaines données ont été exclues (par exemple à cause de la plage temporelle).
    4. Filtre les KPI en fonction des modèles restants (hors données d’entrée).

    Paramètres :
        id_modele_entree (list[str]) : Identifiant des données d’entrée.
        selection_options (dict) : Dictionnaire contenant les options de sélection activées.
        selection_date (tuple) : Plage temporelle sélectionnée par l’utilisateur.
        df_final (pd.DataFrame) : Données complètes à filtrer (entrée + prédictions).
        donnees_kpi (list[dict]) : Liste des indicateurs de performance par modèle.
        liste_modeles_id (list[str]) : Liste de tous les modèles disponibles.
        var_id (str) : Nom de la colonne identifiant les modeles
        id_modele_moyen (list[str]) : Identifiant du modèle moyen.
        choix_unite (str) : Unité de mesure sélectionnée.
        choix_temps (str) : Type de temps sélectionné.

    Résultats retournés :
        - pd.DataFrame : Données filtrées à afficher dans le graphique.
        - pd.DataFrame : Données KPI filtrées à afficher dans le tableau.
        - list[str] : Liste finale des modeles après filtrage.
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
    Génère dynamiquement le titre du graphique ainsi que les étiquettes des axes,
    en fonction des données sélectionnées par l’utilisateur.

    Le titre est construit à partir des éléments cochés :
    - données d'entrée
    - ensemble des prédictions
    - moyenne des prédictions

    Paramètres :
        selection_options (dict) : Dictionnaire contenant les états des options sélectionnées par l'utilisateur.
        choix_temps (str) : Type de temps utilisé pour l'axe des abscisses ('temps horaire' ou 'temps relatif').
        liste_unite (list[str]) : Liste des unités disponibles (non utilisée ici, mais transmise pour cohérence).
        choix_unite (str) : Unité de mesure sélectionnée.

    Résultats retournés :
        - str : Titre complet. (était affiché au dessus du graphe mais plus utilisé)
        - str : Libellé de l’axe des X.
        - str : Libellé de l’axe des Y (incluant l’unité).
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
    Affiche dans l’interface Streamlit deux groupes de cases à cocher :
    - les types de données à exporter (prédictions, métriques),
    - les formats d’exportation (CSV, PDF, PNG).

    Résultat retourné :
        - dict : Dictionnaire contenant deux sous-dictionnaires :
            - "donnees" : {clé: booléen} pour chaque type de donnée sélectionnée.
            - "formats" : {clé: booléen} pour chaque format d’export sélectionné.
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



