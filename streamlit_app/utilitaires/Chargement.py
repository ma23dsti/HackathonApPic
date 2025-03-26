#import streamlit as st
import pandas as pd
import os
import json



# fonction pour charger le fichier
def chargement_donnees():

    # Récupérer le dossier où se trouve Chargement.py
    script_dir = os.path.dirname(__file__)
    # Ajouter chemin vers resultats/donnees_a_la_volee/resultats.json
    #file_path = os.path.join(script_dir, '..', 'resultats/donnees_a_la_volee', 'resultats.json')
    file_path = os.path.join(script_dir, '..', 'resultats/donnees_a_la_volee', 'resultats_augmente.json')
    # Normalisation du chemin
    file_path = os.path.abspath(file_path)
    #chargement fichier
    with open (file_path,'r', encoding='utf-8') as f:
        data=json.load(f)
    

    # creation d'un dataframe pour récuperer les données d'entrée
    entree=data['donnees_entrees']
    df_donnees_entrees=pd.DataFrame({
        'temps horaire':pd.to_datetime(entree['temps_horaire']),
        'temps relatif':pd.Series(entree['temps_relatif']).astype(int),
        'id donnee':'modele_entree',
        'nom donnee':'données d\'entrée',
        'unite mesure':entree['unite_mesure'],
        'format mesure': entree['format_mesure'],
        'valeur':entree['donnees_observees'],
    })


    # récupération des données de prédiction
    prevision=data['resultats']['predictions']
    n=len(prevision['temps_relatif']) #nbre de données de prediction disponibles par modele

    #initialisation des dictionnaires
    donnees_prevision=[]
    donnees_kpi = []
    
    # parcourt sur chaque modele 
    for modele in prevision['modeles']:
        # Parcourt sur les données KPI
        for indicateur, valeur in modele['kpi'].items():
            # construction d'un dictionnaire pour récupérer les données KPI
            donnees_kpi.append({
                'id donnee': modele['id'],
                'nom donnee':modele['nom'],
                'indicateur': indicateur,
                'valeur':valeur})
        #parcourt sur les index de prediction
        for i in range(n):
            #construction d'un dictionnaire donnees_prevision pour récuperer les données sélectionnées selon index prédiction et modele
            donnees_prevision.append({
            'temps horaire':pd.to_datetime(prevision['temps_horaire'][i]),
            'temps relatif': int(prevision['temps_relatif'][i]),
            'id donnee':modele['id'],
            'nom donnee':modele['nom'],
            'unite mesure':entree['unite_mesure'],
            'format mesure': entree['format_mesure'],
            'valeur':modele['donnees_predites'][i]})
            
        
    #fusion des donnees d'entrée et de prévision pour avoir un unique dataframe
    df_entrees_prevision=pd.concat([df_donnees_entrees,pd.DataFrame(donnees_prevision)], ignore_index=True)

    var_id='id donnee'
    var_val='valeur'
    
    #creation d'un dictionnaire pour avoir les combinaisons de mesure+ format possible
    mesure_format=[]
    mesure_format={'unite mesure':entree['unite_mesure'],'format mesure': entree['format_mesure']}
     

    #creation de listes utiles pour filtrer sur le dataframe
    id_modele_moyen=['moyenne'] #rappel Id model moyen (moyenne des prédictions)
    id_modele_entree=['modele_entree'] #rappel Id model des données d'entrée (données observées)
    min_date=df_entrees_prevision['temps horaire'].min().to_pydatetime() # date min du df
    max_date=df_entrees_prevision['temps horaire'].max().to_pydatetime() # date max du df
    min_date_entree=df_donnees_entrees['temps horaire'].min().to_pydatetime() # date min du df
    max_date_entree=df_donnees_entrees['temps horaire'].max().to_pydatetime() # date max du df
    min_date_prevision=pd.DataFrame(donnees_prevision)['temps horaire'].min() # date min du df
    max_date_prevision=pd.DataFrame(donnees_prevision)['temps horaire'].max() # date max du df



    # liste des unités de mesure possible
    liste_unite = df_entrees_prevision['unite mesure'].unique().tolist()

    # récupération liste Id des modeles
    liste_modeles_id = [modele['id'] for modele in prevision['modeles'] if modele['id']!= id_modele_moyen[0] ]
    
    #taille de la liste des modeles wo modele moyen
    nb_modele=len(liste_modeles_id)

    # Nom des colonnes temporelles
    col_temps=[col for col in df_entrees_prevision.columns if 'temps' in col.lower()]


    return df_entrees_prevision, donnees_kpi, col_temps, liste_modeles_id,nb_modele,id_modele_moyen,id_modele_entree,liste_unite,mesure_format, var_id, var_val, min_date, max_date, min_date_entree, max_date_entree, min_date_prevision, max_date_prevision





