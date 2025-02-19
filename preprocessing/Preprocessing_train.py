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

# 5 : Transformer les données :
# L'utilisateur peut choisir les données en bit ou octet par seconde (paramètre type de la fonction, string = 'bit' ou 'octet')
# Si les valeurs sont en 'octet', transformer les valeurs  en bit en multipliant par 8 les valeurs.


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

def preprocess_data(df, ecart_debit_max = 15 , purc_valid_jeu = 0.3, horizon=5):
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
    df = df.groupby(df["Time"]).agg({"debit": "mean"}).reset_index()
    df["debit"] = df["debit"].astype(int)  # Assurer que les valeurs restent des entiers

    print("Vérification des données continues et imputation")
    # Vérification des données continues et imputation
    df = df.sort_values("Time").reset_index(drop=True)
    df["Check"] = 0
    df["Time_Diff"] = df["Time"].diff().dt.total_seconds()

    for i in range(1, len(df)):
        if 1 <= df.loc[i, "Time_Diff"] <= ecart_debit_max:
            df.loc[i, "debit"] = (df.loc[i - 1, "debit"] + df.loc[i, "debit"]) // 2
        elif df.loc[i, "Time_Diff"] > ecart_debit_max:
            df.loc[i, "Check"] = 1

    print(
        f"Nombre d'écart détecté supérieur à {ecart_debit_max} secondes : {df.Check.sum()}."
    )
    print(f"Ecart maximum detecté : {df.Time_Diff.max()} secondes.")

    print("Vérification du pourcentage de valeurs avec Check=1")
    # Vérification du pourcentage de valeurs avec Check=1
    if df["Check"].mean() > purc_valid_jeu:
        raise ValueError(
            "Les données ne sont pas correctement agrégées ou contiennent trop d'écarts, plus de {purc_valid_jeu}% de données avec des écart > à {ecart_debit_max} secondes."
        )


    print("Reshape des données")
    # Reshape des données
    valid_sequences = []
    for i in range(len(df) - (shape + horizon) + 1):
        segment = df.iloc[i : i + shape + horizon]
        if segment["Check"].sum() == 0:
            valid_sequences.append(segment["debit"].values)

    if not valid_sequences:
        raise ValueError("Aucune séquence valide trouvée après filtrage")

    reshaped_data = np.array(valid_sequences)
    train_data = reshaped_data[:, :shape]
    valid_data = reshaped_data[:, shape:]

    return train_data, valid_data

def main():
    parser = argparse.ArgumentParser(description="Prétraitement des données de séries temporelles")
    parser.add_argument("path", type=str, help="Chemin du fichier CSV")
    parser.add_argument("--data_type", type=str, default="bit", choices=["bit", "octet"], help="Type de données: 'bit' ou 'octet'")
    parser.add_argument("--data_type_strength", type=str, default="M", help="Paramètre de force des données")
    parser.add_argument("--ecart_debit_max", type=int, default=10, help="Seuil de détection des écarts en secondes")
    parser.add_argument("--purc_valid_jeu", type=float, default=0.1, help="Seuil du pourcentage d'écarts acceptés (entre 0 et 1)")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5, 10, 60, 300], help="Valeur d'horizon pour le reshape")
    
    args = parser.parse_args()
    
    # Charger les données
    df, data_type, data_type_strength = load_data(args.path, args.data_type, args.data_type_strength)
    
    # Prétraiter les données
    train_data, valid_data = preprocess_data(df, args.ecart_debit_max, args.purc_valid_jeu, data_type, args.horizon)
    
    # Sauvegarde des fichiers
    base_path = os.path.splitext(args.path)[0]
    np.savetxt(f"{base_path}_X.csv", train_data, delimiter=",", fmt="%d")
    np.savetxt(f"{base_path}_y.csv", valid_data, delimiter=",", fmt="%d")
    
    print(f"Fichiers sauvegardés: {base_path}_X.csv et {base_path}_y.csv")

if __name__ == "__main__":
    main()
