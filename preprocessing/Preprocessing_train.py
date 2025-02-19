# Import des librairies 
import pandas as pd
import numpy as np
import argparse
import os



## Preprocess de données

# Chargement des données en entrée sous deux colonnes 'Time' et 'debit'. Transormations à effectuer*

# 1 : Vérifier que je possède bien des données sous format date et int64 :
# Si ce n'est pas le cas, essayer de changer les formats avec le format de date YYYY-MM-dd hh:mm:ss.
# Si impossible on raise une error : 'les données doivent être sous le format  YYYY-MM-dd hh:mm:ss et une valeur entière'

# 2 :  Agréger les valeurs si plusieurs appartiennent à la même seconde

# 3 : Vérifier que les données soit continnues :
# Il ne faut pas qu'il y ait des secondes vides entres deux lignes.
# Si il manque jusqu'à x secondes de valeur manquantes, on impute les données par la moyenne de la valeurs précédente et de la valeur suivante.
# Si il y a plus de x secondes d'écart en la ligne précédente et celle-là (a étant un paramètre de la fonction un entier > 0) , mettre dans une nouvelle colonne check la valeur 1.

# 4 : Vérifier qu'il n'y a pas trop d'écart :
# Si la colonne Check contient plus de x% de valeur égales à 1 (x étant une données d'entrée par l'utilisateur, compris entre 0 et 1) raise une nouvelle erreur : 'les données ne sont pas aggrégées à la seconde ou contiennent des valeures avec trop d'écart, veuillez essayer un nouveau jeu de données.'

# 6 : Reshape les données :
# Un paramètre horizon sera en entrée avec 5 valeurs possibles  : 1 seconde, 5 secondes, 10 secondes, 60 secondes ou 300 secondes.
# Un mapping sera définit en amont de la fonction selon chaque valeurs d'horizon une valeur shape sera définit :
# valeur horizon:shape   1: 12, 5:60, 10:90, 60:120, 300:190.
# Selon la valeur de shape il faudra reshape les données avec la valeur de shape+horizon comme nombre de colonne.
# Attention il faudra s'assurer que les données reshape au sein d'une même ligne ne contiennent aucune valeur de check égale à 1.
# Si une valeur check =1 est trouvé, il faudra trouver une oublier cette séquence et trouver une prochaine séquence possible sans aucun check = 1

# 7 : Split les données : Enfin il faudra que ma fonction sépare les données en 2 jeux train et valid
# train : les shape premières colonnes du jeu de données
# valid : les horizon dernières colonnes  de mon jeu de données


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

def preprocess_data(df, ecart_debit_max = 30 , purc_valid_jeu = 0.3, horizon=5):
    # Mapping horizon to shape
    horizon_mapping = {1: 12, 5: 60, 10: 90, 60: 120, 300: 190}
    if horizon not in horizon_mapping:
        raise ValueError("Horizon must be one of [1, 5, 10, 60, 300]")
    shape = horizon_mapping[horizon]

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

    print("Vérification des données continues et imputation")
    # Vérification des données continues et imputation
    df = df.sort_values("Time").reset_index(drop=True)
    df["Time_Diff"] = df["Time"].diff().dt.total_seconds()
    df.Time_Diff = df.Time_Diff.fillna(1)

    # Quand on a des ecart inférieur à ecart debit max :
    lines = df.loc[(df.Time_Diff <= ecart_debit_max) & (df.Time_Diff > 1), :].index
    previous_lines = lines - 1

    bad_time_diff = np.array(df.loc[df.index.isin(lines), "Time_Diff"]).astype(int)
    last_time = np.array(df.loc[df.index.isin(lines), "Time"])
    first_time = np.array(df.loc[df.index.isin(previous_lines), "Time"])
    last_debit = np.array(df.loc[df.index.isin(lines), "debit"])
    first_debit = np.array(df.loc[df.index.isin(previous_lines), "debit"])

    timedelta = np.timedelta64(1, "s")  # 1 second
    for (first_time,first_debit,last_time,last_debit,nb_new_lines,) in zip(first_time, first_debit, last_time, last_debit, bad_time_diff):
        # On enleve les extremites
        first_time = first_time + timedelta
        last_time = last_time - timedelta
        add_time = pd.date_range(
            start=first_time,
            end=last_time,
            freq="s",
        )
        add_debit = np.linspace(first_debit, last_debit, num=nb_new_lines - 1)
        size = len(add_time)
        new_data = {
            "Time": add_time,
            "debit": add_debit,
            "Time_Diff": [1] * size,
        }
        new_df = pd.DataFrame(new_data)
        df = pd.concat([df, new_df], ignore_index=True)

    df = df.sort_values("Time").reset_index(drop=True)
    # On actualise les Time_Diff
    df["Time_Diff"] = df["Time"].diff().dt.total_seconds()

    # Quand les ecart sont trop grand :
    df["Check"] = (df["Time_Diff"] > 1).astype(int)
     
    print(
        f"Nombre d'écart détecté supérieur à {ecart_debit_max} secondes : {df.Check.sum()}."
    )
    print(f"Ecart maximum detecté : {df.Time_Diff.max()} secondes.")

    print("Vérification du pourcentage de valeurs avec Check=1")
    # Calculer intervalle de temps du jeu de données
    difference = df.Time.max() - df.Time.min()
    # Obtenir la différence en secondes
    secondes = difference.total_seconds()
    missing_info = np.sum(df["Check"] * df["Time_Diff"]) / secondes

    if missing_info > purc_valid_jeu:
        raise ValueError(
            "Les données ne sont pas correctement agrégées ou contiennent trop d'écarts, plus de {purc_valid_jeu}% de données avec des écart > à {ecart_debit_max} secondes."
        )

    print("Reshape des données")
    # Reshape des données
    valid_sequences = []
    size = shape + horizon
    for i in range(0, len(df), size):
        segment = df.iloc[i : i + shape + horizon]
        if segment["Check"].sum() == 0:
            valid_sequences.append(segment["debit"].values)
    preprocess_data = pd.DataFrame(valid_sequences)

    X = preprocess_data.iloc[:, :shape]
    y = preprocess_data.iloc[:, shape:]
    print("Done")
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Prétraitement des données de séries temporelles")
    parser.add_argument("path", type=str, help="Chemin du fichier CSV")
    parser.add_argument("--ecart_debit_max", type=int, default=30, help="Seuil de détection des écarts en secondes")
    parser.add_argument("--purc_valid_jeu", type=float, default=0.3, help="Seuil du pourcentage d'écarts acceptés (entre 0 et 1)")
    parser.add_argument("--horizon", type=int, default=5, choices=[1, 5, 10, 60, 300], help="Valeur d'horizon pour le reshape")
    
    args = parser.parse_args()
    
    # Charger les données
    df = load_data(args.path)
    
    # Prétraiter les données
    X, y = preprocess_data(df, args.ecart_debit_max, args.purc_valid_jeu, args.horizon)
    
    # Sauvegarde des fichiers
    base_path = os.path.splitext(args.path)[0]
    X.to_csv(f"{base_path}_X.csv", index=False,header=False) 
    y.to_csv(f"{base_path}_y.csv", index=False,header=False) 
    
    print(f"Fichiers sauvegardés: {base_path}_X.csv et {base_path}_y.csv")

if __name__ == "__main__":
    main()
