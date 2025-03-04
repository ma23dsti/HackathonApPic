# Import des librairies 
import pandas as pd
import numpy as np
import argparse
import os



## Preprocess de données predict 

# Chargement des données en entrée sous deux colonnes 'Time' et 'debit'. Transormations à effectuer*

# 1 : Vérifier que je possède bien des données sous format date et int64 :
# Si ce n'est pas le cas, essayer de changer les formats avec le format de date YYYY-MM-dd hh:mm:ss.
# Si impossible on raise une error : 'les données doivent être sous le format  YYYY-MM-dd hh:mm:ss et une valeur entière'

# 2 :  Agréger les valeurs si plusieurs appartiennent à la même seconde

# 3 : Vérifier que les données soit continnues :
# Il ne faut pas qu'il y ait des secondes vides entres deux lignes.
# Si il manque jusqu'à x secondes de valeur manquantes, on impute les données par la moyenne de la valeurs précédente et de la valeur suivante.
# Si il y a plus de x secondes d'écart en la ligne précédente et celle-là (a étant un paramètre de la fonction un entier > 0) , mettre dans une nouvelle colonne check la valeur 1.

# 5 : Vérifier qu'il n'y a pas trop d'écart :
# Si la colonne Check contient plus de x% de valeur égales à 1 (x étant une données d'entrée par l'utilisateur, compris entre 0 et 1) raise une nouvelle erreur : 'les données ne sont pas aggrégées à la seconde ou contiennent des valeures avec trop d'écart, veuillez essayer un nouveau jeu de données.'

# 6 : Reshape les données :
# Selon  l'horizon de prédiction choisi, un mapping sera définit en amont de la fonction pour définir la valeur de la shape :
# valeur horizon:shape   1: 12, 5:60, 10:90, 60:120, 300:190.
# Selon la valeur de shape il faudra reshape les données avec la valeur shape comme nombre de colonne.
# Attention il faudra s'assurer que les données reshape au sein d'une même ligne ne contiennent aucune valeur de check égale à 1.
# Si une valeur check =1 est trouvé, il faudra trouver une oublier cette séquence et trouver une prochaine séquence possible sans aucun check = 1


# Faire les opérations pour le jeu de données et retourner un fichier x_test

# Vérifier que les données de test soit différentes des données de train  

def load_data(path):
    # Lecture du fichier avec détection automatique du séparateur et de l'en-tête
    df = pd.read_csv(path, sep=None, engine='python')
    
    # Vérifier que le fichier contient exactement deux colonnes
    if df.shape[1] != 2:
        raise ValueError("Le fichier doit contenir exactement deux colonnes")
    
    # Renommer les colonnes en 'Time' et 'debit'
    df.columns = ['Time', 'debit']
    
    return df


# Pour check les trou avec purc_valid_jeu , se baser sur les time diff et pas que sur le nombre de check et faire check * time_diff / temps total du dataset 

def preprocess_data(df, ecart_debit_max=30, purc_valid_jeu=0.4, horizon=5 , shape = 60):

    print("Vérification et conversion des types")
    # Vérification et conversion des types
    try:
        df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S")
    except Exception:
        raise ValueError("Les données doivent être sous le format YYYY-MM-dd hh:mm:ss")

    if not np.issubdtype(df["debit"].dtype, np.integer):
        raise ValueError("Les données doivent être sous forme d'entiers")

    print("Agréger les valeurs si plusieurs appartiennent à la même seconde")
    # Agréger les valeurs si plusieurs appartiennent à la même seconde
    df = df.groupby(df["Time"]).agg({"debit": "sum"}).reset_index()
    df["debit"] = df["debit"].astype(int)  # Assurer que les valeurs restent des entiers

    # S'assurer que les données soient continnues et imputation
    df = df.sort_values("Time").reset_index(drop=True)
    df["Time_Diff"] = df["Time"].diff().dt.total_seconds()
    df.Time_Diff = df.Time_Diff.fillna(1)

    # Quand on a des ecart inférieur à ecart debit max :
    lines = df.loc[(df.Time_Diff <= ecart_debit_max) & (df.Time_Diff > 1), :].index
    previous_lines = lines - 1
    print("Imputation des valeurs manquantes : ", len(lines), "imputations à faire")
    # On récupères les débits de la premiere et la dernièere ligne entre chaque imputation et l'écart de temps entre les lignes
    bad_time_diff = np.array(df.loc[df.index.isin(lines), "Time_Diff"]).astype(int)
    last_time = np.array(df.loc[df.index.isin(lines), "Time"])
    first_time = np.array(df.loc[df.index.isin(previous_lines), "Time"])
    last_debit = np.array(df.loc[df.index.isin(lines), "debit"])
    first_debit = np.array(df.loc[df.index.isin(previous_lines), "debit"])

    timedelta = np.timedelta64(1, "s")  # 1 second
    for (
        first_time,
        first_debit,
        last_time,
        last_debit,
        nb_new_lines,
    ) in zip(first_time, first_debit, last_time, last_debit, bad_time_diff):
        # On enlève les valeurs des bords qui elles ne sont pas à imputer
        first_time = first_time + timedelta
        last_time = last_time - timedelta
        # On rajoute autant de ligne que de seconde manquantes
        add_time = pd.date_range(
            start=first_time,
            end=last_time,
            freq="s",
        )

        # Imputation linéaire des débits entre première et dernière valeur
        add_debit = np.linspace(first_debit, last_debit, num=nb_new_lines - 1)
        size = len(add_time)
        new_data = {
            "Time": add_time,
            "debit": add_debit,
            "Check": [0] * size,
            "Time_Diff": [1] * size,
        }

        # On rajoute nos nouvelles données à notre dataframe
        new_df = pd.DataFrame(new_data)
        df = pd.concat([df, new_df], ignore_index=True)

    df = df.sort_values("Time").reset_index(drop=True)
    print("Imputation terminé")
    df["Time_Diff"] = df["Time"].diff().dt.total_seconds()

    # Quand les ecart sont trop grand :
    df["Check"] = (df["Time_Diff"] > 1).astype(int)
    print(
        f"Nombre d'écart détecté supérieur à {ecart_debit_max} secondes : {df.Check.sum()}."
    )
    print(f"Ecart maximum detecté : {df.Time_Diff.max()} secondes.")

    # Vérifier qu'il n'y pas trop de temps manquants dans le jeu de données
    # Calculer la différence
    difference = df.Time.max() - df.Time.min()

    # Obtenir la différence en secondes
    secondes = difference.total_seconds()

    missing_info = np.sum(df["Check"] * df["Time_Diff"]) / secondes

    print(f"{missing_info:.2f}% de temps manquant dans le jeu de données")
    if missing_info > purc_valid_jeu:
        raise ValueError(
            "Les données ne sont pas correctement agrégées ou contiennent trop d'écarts, plus de {purc_valid_jeu:.2f}% de données avec des écart > à {ecart_debit_max} secondes."
        )

    # Reshape des données
    print("Reshape des données")
    print("horizon de prédiction voulu : ",horizon, " secondes")
    print("Shape appliquée : ",shape)
    valid_sequences = []
    size = shape
    for i in range(0, len(df), shape):
        segment = df.iloc[i : i + size]
        if segment["Check"].sum() == 0:
            valid_sequences.append(segment["debit"].values)
    preprocess_data = pd.DataFrame(valid_sequences)

    # On enleve la derniere ligne si elle n'est pas complète
    clean_data = preprocess_data.dropna()
    print("Reshape des données terminées")

    return clean_data



def check_similarity(x_train: pd.DataFrame, x_predict: pd.DataFrame, threshold: float = 50.0): 
    #Fonction qui permet de vérifier que x_train et x_valid sont différents 
    # Convertir les DataFrames en ensembles de tuples (pour une comparaison rapide)
    train_set = set(map(tuple, x_train.to_numpy()))
    predict_set = set(map(tuple, x_predict.to_numpy()))

    # Trouver les lignes communes
    common_rows = predict_set.intersection(train_set)

    # Calculer le pourcentage de similarité
    similarity_percentage = (len(common_rows) / len(x_predict)) * 100

    print(f"Pourcentage de similarité : {similarity_percentage:.2f}%")

    # Lever une erreur si le seuil est dépassé
    if similarity_percentage > threshold:
        raise ValueError(f"Le pourcentage de similarité ({similarity_percentage:.2f}%) dépasse {threshold}% !")


def main():
    parser = argparse.ArgumentParser(description="Prétraitement des données de séries temporelles")
    parser.add_argument("path", type=str, help="Chemin du fichier CSV")
    parser.add_argument("--ecart_debit_max", type=int, default=30, help="Seuil de détection des écarts en secondes")
    parser.add_argument("--purc_valid_jeu", type=float, default=0.3, help="Seuil du pourcentage d'écarts acceptés (entre 0 et 1)")
    parser.add_argument("--horizon", type=int, default=5, choices=[1, 5, 10, 60, 300], help="Valeur d'horizon du nombre de seconde à prédire")
    parser.add_argument("path_train", type=str, help="Chemin du fichier de train")
    
    args = parser.parse_args()
    
        # Mapping horizon to shape
    horizon_mapping = {1: 12, 5: 60, 10: 90, 60: 120, 300: 190}

    if args.horizon not in horizon_mapping:
        raise ValueError("Horizon doit valoir 1, 5, 10, 60 ou 300 secondes")
    shape = horizon_mapping[args.horizon]
    # Vérification des paramètres : 

    if args.ecart_debit_max < 0:
        raise ValueError("ecart_debit_max doit etre une valeur positive ou nulle")

    if args.purc_valid_jeu > 1 or args.purc_valid_jeu <= 0:
        raise ValueError("purc_valid_jeu doit etre compris entre 0 et 1")

    # Charger les données
    df = load_data(args.path)

    # Prétraiter les données de prédiction
    print("Preprocess des données de prédiction") 
    X_predict = preprocess_data(df, args.ecart_debit_max, args.purc_valid_jeu, args.horizon, shape)

    print("Données de prediction : ",len(X_predict))


    # Vérifier si validation et train ont des données trop similaire ou non 50%

    # Charger le jeu de train
    X_train = load_data(args.path_train) 
    print("Vérification des similarité du jeu de train et de prédiction ")
    check_similarity(X_train, X_predict)

    # Sauvegarde des fichiers
    base_path = os.path.splitext(args.path)[0]

    name_file_x_predict = f"{base_path}_x_predict_o{shape}_p{args.horizon}.csv"

    X_predict.to_csv(name_file_x_predict, index=False,header=False)
    
    print(f"Fichiers sauvegardés: {name_file_x_predict} ")

if __name__ == "__main__":
    main()
