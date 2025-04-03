# Import des librairies 
import pandas as pd
import numpy as np
import argparse
import os



## Preprocess de données predict 


# V0 : Seulement des checks à vérifier : 

# Chargement des données en entrée sous deux colonnes 'Time' et 'debit'. Transormations à effectuer*

# Vérifier que je possède bien des données sous format date et int64 :

# Vérifier que les données soit bien avec une colonne timestamp et une colonne débit

# S'assurer qu'on ait bien une valeur par seconde sans discontinuité

# Vérifier que le nombre d'input soit égale à la shape choisi par l'utilisateur 

# Vérifier que les données de test soit différentes des données de train 

# V1 (Not done yet): Preprocess les données un peu : 

# Si plusieurs données à la seconde, aggréger les valeurs

# Si discontinuité faible, remplir les trous 

# Si la taille des données de prédiction est plus grande que celle choisi, couper pour ne garder que les x derniere valeur pour la prédiction 


def check_similarity(x_train: pd.DataFrame, x_test: pd.DataFrame,  threshold: float = 50.0) -> None: 
    """
    Fonction qui permet de vérifier que x_train et x_test sont différents 
    Convertir les DataFrames en ensembles de tuples (pour une comparaison rapide)

    """
    train_set = set(map(tuple, x_train.to_numpy()))
    test_set = set(map(tuple, x_test.to_numpy()))

    # Trouver les lignes communes
    common_rows = test_set.intersection(train_set)

    # Calculer le pourcentage de similarité
    similarity_percentage = (len(common_rows) / len(x_test)) * 100

    print(f"Pourcentage de similarité : {similarity_percentage:.2f}%")

    # Lever une erreur si le seuil est dépassé
    if similarity_percentage > threshold:
        raise ValueError(f"Le pourcentage de similarité ({similarity_percentage:.2f}%) dépasse {threshold}% !")
    pass


def check_data(test_data: pd.DataFrame, train_data: pd.DataFrame, horizon: int, preprocess_dir : str) -> bool:
    """
    Vérifie l'intégrité des données de test.
    :param test_data: DataFrame avec les colonnes 'Time' et 'debit'
    :param train_data: DataFrame avec les colonnes 'Time' et 'debit'
    :param horizon: Nombre d'inputs attendu
    :param preprocess_dir : Endroit où sauvegarder les données de test
    :return: True si toutes les vérifications passent, sinon une exception est levée
    """

    # Mapping horizon to shape
    horizon_mapping = {1: 12, 5: 60, 30: 300, 60: 400, 300: 500}
    shape = horizon_mapping[horizon]
    if horizon not in horizon_mapping:
        raise ValueError("Horizon doit valoir 1, 5, 30, 60 ou 300")

    # Vérification des paramètres : 
    if not isinstance(test_data, pd.DataFrame):
        raise TypeError("test_data doit être un DataFrame pandas.")
    if not isinstance(train_data, pd.DataFrame):
        raise TypeError("train_data doit être un DataFrame pandas.")
    
    # Vérification des colonnes
    expected_columns = {'Time', 'debit'}
    if set(test_data.columns) != expected_columns:
        raise ValueError(f"Les colonnes doivent être {expected_columns}, mais on a {set(test_data.columns)}")
    
    print("Vérification et conversion des types")
    # Vérification et conversion des types
    try:
        test_data["Time"] = pd.to_datetime(test_data["Time"], format="%Y-%m-%d %H:%M:%S")
    except Exception:
        raise ValueError("Les données doivent être sous le format YYYY-MM-dd hh:mm:ss")

    if not np.issubdtype(test_data["debit"].dtype, np.integer):
        raise ValueError("Les données doivent être sous forme d'entiers")

    # S'assurer que les données soient continnues et imputation
    test_data = test_data.sort_values("Time").reset_index(drop=True)
    test_data["Time_Diff"] = test_data["Time"].diff().dt.total_seconds()
    test_data.Time_Diff = test_data.Time_Diff.fillna(1)
    if (len(test_data.loc[test_data.Time_Diff > 1, 'Time_Diff']) > 0) : 
        raise ValueError("Les données ne sont pas aggrégés à la seconde")
    

    # Vérification du nombre d'inputs
    if len(test_data) != shape:
        raise ValueError(f"Le nombre d'inputs ({len(test_data)}) ne correspond pas à l'horizon choisi ({shape})")
    

    #Reshape 
    df_test = pd.DataFrame(test_data['debit'].values.reshape(1, -1))

    #Vérifier que les données de test soit différentes des données de train 
    check_similarity(train_data, df_test)

    print("Les données sont valides ! ")

    # Sauvegarde des fichiers
    print("Sauvegarde du fichier preprocess :")

    name_file_x_test = f"{preprocess_dir}_x_test_o{shape}_p{horizon}.csv"

    df_test.to_csv(name_file_x_test, index=False,header=False)
    
    print(f"Fichiers sauvegardés: {name_file_x_test} ")

    return True 

