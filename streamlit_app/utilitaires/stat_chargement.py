import pandas as pd
import os
import json
import streamlit as st




import json

def charger_fichier_json():
    """
    Charge le fichier JSON contenant les données d'entrée et de prévision.

    Retourne:
        dict : Dictionnaire Python représentant le contenu du fichier JSON.

    Exceptions:
        FileNotFoundError : Si le fichier JSON est introuvable à l'emplacement spécifié.
        ValueError : Si le contenu du fichier n'est pas un dictionnaire.
    """

    # Récupérer le dossier où se trouve Chargement.py
    script_dir = os.path.dirname(__file__)
    # Ajouter chemin vers resultats/donnees_a_la_volee/resultats.json
    file_path = os.path.join(script_dir, '..', 'resultats/donnees_a_la_volee', 'resultats.json')
    
    # Normalisation du chemin
    file_path = os.path.abspath(file_path)

    # Vérifier si le fichier existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier JSON est introuvable à l’emplacement : {file_path}")

    #chargement fichier
    with open (file_path,'r', encoding='utf-8') as f:
        data=json.load(f)

    #vérifie que le contenu est bien un dictionnaire
    if not isinstance(data, dict):
        raise ValueError("Le fichier JSON n'a pas le format attendu (dictionnaire JSON).")

    return data



def extraire_donnees_entree(data):
    """
    Crée un DataFrame à partir des données d'entrée présentes dans le JSON.
    Gère deux unités de mesure différentes et retourne les métadonnées associées.

    Paramètre :
        data (dict): Dictionnaire chargé depuis le fichier JSON.

    Retourne:
        tuple:
            - pd.DataFrame: Données d'entrée combinées au format tabulaire.
            - dict: Données de la clé 'donnees_entrees' issues du JSON.
            - list[str]: Liste des unités de mesure utilisées.
            - list[str]: Liste contenant l'identifiant unique pour les données d'entrée.

    Exceptions:
        KeyError: Si une clé obligatoire est manquante dans le dictionnaire JSON.
        ValueError: Si des unités sont vides ou si les longueurs des séries ne correspondent pas.
    """
    if 'donnees_entrees' not in data:
        raise KeyError("Clé 'donnees_entrees' manquante dans le JSON.")
    
    entree = data['donnees_entrees']
    
    # Vérification des clés principales
    entree_cles = ['temps_horaire', 'temps_relatif', 'unite_mesure', 'unite_mesure_2', 'format_mesure', 'donnees_observees']
    for key in entree_cles:
        if key not in entree:
            raise KeyError(f"Clé obligatoire manquante dans 'donnees_entrees' : '{key}'")

    # Vérification que les unités de mesure ne sont pas vides
    liste_unite = [entree['unite_mesure'], entree['unite_mesure_2']]
    if not all(liste_unite):
        raise ValueError("Une ou plusieurs unités de mesure sont vides ou nulles dans 'donnees_entrees'.")

    # Récupération des temps et format
    temps_horaire = pd.to_datetime(entree['temps_horaire'])
    temps_relatif = pd.Series(entree['temps_relatif']).astype(int)
    #format_mesure = entree['format_mesure']
    
    # Récupération des données observées
    donnees = entree['donnees_observees']
    valeurs1 = donnees['en_unite_mesure']
    valeurs2 = donnees['en_unite_mesure_2']

    if not (len(temps_horaire) == len(temps_relatif) == len(valeurs1) == len(valeurs2)):
        raise ValueError("Les longueurs de 'temps_horaire', 'temps_relatif' et des séries de données observées ne correspondent pas.")
    
    n = len(temps_horaire)

    id_modele_entree = ['0_entree']
    
    # Création du DataFrame pour unite_mesure
    df1 = pd.DataFrame({
        'temps horaire': temps_horaire,
        'temps relatif': temps_relatif,
        'id donnee': id_modele_entree * n,
        'nom donnee': ['données d\'entrée'] * n,
        'unite mesure': [liste_unite[0]] * n,
       # 'format mesure': [format_mesure] * n,
        'valeur': valeurs1
    })

    # Création du DataFrame pour unite_mesure_2
    df2 = pd.DataFrame({
        'temps horaire': temps_horaire,
        'temps relatif': temps_relatif,
        'id donnee': ['0_entree'] * n,
        'nom donnee': ['données d\'entrée'] * n,
        'unite mesure': [liste_unite[1]] * n,
        #'format mesure': [format_mesure] * n,
        'valeur': valeurs2
    })

    # Fusion des deux DataFrames
    df_donnees_entrees = pd.concat([df1, df2], ignore_index=True)

    return df_donnees_entrees, entree, liste_unite, id_modele_entree



def extraire_donnees_prediction_et_kpi(data, entree):
    """
    Extrait les données de prédiction et les indicateurs de performance (KPI) 
    pour chaque modèle contenu dans le fichier JSON.

    Cette fonction gère deux séries de valeurs correspondant aux deux unités de mesure
    et vérifie la cohérence des informations avant construction du df final.

    Paramètres :
        data (dict) : Dictionnaire chargé à partir du fichier JSON.
        entree (dict) : Partie du JSON correspondant aux données d'entrée, contenant les unités de mesure.

    Résultats retournés :
        - pd.DataFrame : Données de prédiction pour chaque modèle et chaque unité de mesure.
        - list[dict] : Liste des KPI pour chaque modèle, avec identifiant, nom et valeur de chaque indicateur.
        - dict : Partie du JSON correspondant à la section 'predictions'.

    Exceptions levées :
        - KeyError : Si une ou plusieurs clés attendues sont absentes dans la structure JSON.
        - ValueError : Si les longueurs des séries de temps ou des valeurs ne correspondent pas.
        - TypeError : Si les KPI ne sont pas sous forme de dictionnaire.
    """
     
    if 'resultats' not in data or 'predictions' not in data['resultats']:
        raise KeyError("Clé 'resultats > predictions' manquante dans le fichier JSON.")
    
    prediction = data['resultats']['predictions']
    n_prediction = len(prediction['temps_horaire'])

    if 'modeles' not in prediction:
        raise KeyError("Clé 'modeles' manquante dans la section 'predictions'.")

    prediction_cles = ['temps_horaire', 'temps_relatif']
    for key in prediction_cles:
        if key not in prediction:
            raise KeyError(f"Clé '{key}' manquante dans les prédictions.")
    
    np_tr = len(prediction['temps_relatif'])
    if np_tr != n_prediction:
        raise ValueError("Les longueurs de 'temps_horaire' et 'temps_relatif' dans les prédictions ne correspondent pas.")

    donnees_prediction = []
    donnees_kpi = []
    nb_kpi_ref = None

    for modele in prediction['modeles']:
        if 'donnees_predites' not in modele:
            raise KeyError(f"Clé 'donnees_predites' manquante dans le modèle {modele.get('id', 'inconnu')}.")

        donnees_predites = modele['donnees_predites']
        valeurs1 = donnees_predites['en_unite_mesure']
        valeurs2 = donnees_predites['en_unite_mesure_2']

        if len(valeurs1) != n_prediction or len(valeurs2) != n_prediction:
            raise ValueError(f"Le modèle {modele.get('id', 'inconnu')} contient un nombre incorrect de prédictions.")

        for i in range(n_prediction):
            # Série en unite_mesure
            donnees_prediction.append({
                'temps horaire': pd.to_datetime(prediction['temps_horaire'][i]),
                'temps relatif': int(prediction['temps_relatif'][i]),
                'id donnee': modele['id'],
                'nom donnee': modele['nom'],
                'unite mesure': entree['unite_mesure'],
               # 'format mesure': entree['format_mesure'],
                'valeur': round(valeurs1[i],1)
            })
            # Série en unite_mesure_2
            donnees_prediction.append({
                'temps horaire': pd.to_datetime(prediction['temps_horaire'][i]),
                'temps relatif': int(prediction['temps_relatif'][i]),
                'id donnee': modele['id'],
                'nom donnee': modele['nom'],
                'unite mesure': entree['unite_mesure_2'],
               # 'format mesure': entree['format_mesure'],
                'valeur': round(valeurs2[i], 1)
            })

        if 'kpi' not in modele:
            raise KeyError(f"Clé 'kpi' manquante dans le modèle {modele.get('id', 'inconnu')}.")
        if not isinstance(modele['kpi'], dict):
            raise TypeError(f"Les KPI du modèle {modele.get('id', 'inconnu')} doivent être un dictionnaire.")

        nb_kpi = len(modele['kpi'])
        if nb_kpi_ref is None:
            nb_kpi_ref = nb_kpi
        elif nb_kpi != nb_kpi_ref:
            raise ValueError(f"Le modèle {modele.get('id', 'inconnu')} contient {nb_kpi} KPI, attendu : {nb_kpi_ref}.")

        for indicateur, valeur in modele['kpi'].items():
            donnees_kpi.append({
                'id donnee': modele['id'],
                'nom donnee': modele['nom'],
                'indicateur': indicateur,
                'valeur': round(valeur, 4)
            })

    df_prediction = pd.DataFrame(donnees_prediction)
    return df_prediction, donnees_kpi, prediction





def fusionner_et_convertir(df_entree, df_prediction):
    """
    Fusionne les données d'entrée et de prédiction en un seul tableau.

    Cette fonction vérifie que les deux jeux de données ne sont pas vides,
    puis les concatène dans un même DataFrame.

    Paramètres :
        df_entree (pd.DataFrame) : Données entrée (observées initiales), extraites du fichier JSON.
        df_prediction (pd.DataFrame) : Données de prédiction issues des modèles et extraites du fichier JSON.

    Résultat retourné :
        - pd.DataFrame : Jeu de données combiné contenant les données d'entrée
                         et les prédictions pour les deux unités de mesure.

    Exception levée :
        - ValueError : Si l’un des deux tableaux est vide (aucune donnée à fusionner).
    """
    
    if df_entree.empty or df_prediction.empty:
        raise ValueError("Les données d’entrée ou de prévision sont vides, impossible de les fusionner.")

    # Fusion des données d'entrée et de prévision dans un seul DataFrame
    df_final = pd.concat([df_entree, df_prediction], ignore_index=True)

   
    return df_final


def obtenir_info_metadata(df_final, df_donnees_entrees, df_prediction, prediction, entree):
    """
    Extrait les métadonnées nécessaires pour l'affichage et l'analyse des résultats.

    Cette fonction permet de récupérer :
    - les colonnes temporelles à utiliser pour l’axe des abscisses,
    - les identifiants des modèles (hors modèle de moyenne),
    - le nombre de modèles,
    - l'identifiant du modèle moyen,
    - l’unité de mesure par défaut,
    - les colonnes utiles pour les regroupements ou les filtrages.

    Paramètres :
        df_final (pd.DataFrame) : Données fusionnées contenant entrées et prédictions.
        df_donnees_entrees (pd.DataFrame) : Données d'entrée .
        df_prediction (pd.DataFrame) : Données de prédiction.
        prediction (dict) : Partie du JSON contenant les prédictions.
        entree (dict) : Partie du JSON contenant les données d'entrée.

    Résultats retournés :
        - list[str] : Liste des noms de colonnes contenant une information temporelle.
        - list[str] : Liste des identifiants des modèles (hors modèle moyen).
        - int : Nombre total de modèles (hors modèle moyen).
        - list[str] : Identifiant du modèle de moyenne (['moyenne'] - fixé dans le json).
        - str : Unité de mesure par défaut (dans le json unite_mesure dans les 'donnees_entrees' 
        correspond à l'unité sélectionné dans la page d'accueil).
        - str : Nom de la colonne contenant les identifiants des données ('id donnee').
        - str : Nom de la colonne contenant les valeurs numériques ('valeur').
    """
    # création de listes utiles pour filtrer sur le dataframe
    id_modele_moyen = ['moyenne'] # rappel Id modèle moyen (moyenne des prédictions)
    

    # récupération liste ID des modèles, hors modèle moyen
    liste_modeles_id = [modele['id'] for modele in prediction['modeles'] if modele['id'] != id_modele_moyen[0]]
    nb_modele = len(liste_modeles_id) # taille de la liste

    # liste des unités de mesure possible
    unite_mesure_defaut =entree['unite_mesure']
    

    # nom des colonnes temporelles (contenant "temps")
    col_temps = [col for col in df_final.columns if 'temps' in col.lower()]
    
    # nom des variables cibles pour affichage et regroupement
    var_id = 'id donnee'
    var_val = 'valeur'


    return (col_temps,liste_modeles_id,nb_modele,id_modele_moyen,unite_mesure_defaut,var_id,var_val)

   


