# Plateforme de Prédiction de Trafic Réseau

## Prérequis

- Python 3.11
- Anaconda

## Installation

Créez un environnement virtuel avec Anaconda :
    ```bash
    conda env create -n hackathon_env --file hackathon_env.yml
    conda activate hackathon_env
    ```

## Lancer l'application

### Flask

Pour lancer l'application Flask :
```bash
python app.py
```

### Streamlit

Pour lancer l'application Streamlit :
```bash
streamlit run streamlit_app/1_🏠_Accueil.py
```

ou si vous avez une erreur :

```bash
python -m streamlit run streamlit_app/1_🏠_Accueil.py
```
Remplissez les options/paramètres que vous souhaitez puis validez avec le boutons.
Le menu relatif à ce que vous souhaitez faire apparaitra.
Utilisez la barre latérale de Streamlit pour naviguer entre les pages.

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus d’informations.