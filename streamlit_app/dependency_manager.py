import streamlit as st

# Dictionnaire des dépendances pour chaque choix de modèle
dependencies_by_choice = {
    0: {
        "Dépot et Validation des Données": ["valid_acceuil"],
        "Prédictions": ["valid_depot_donnees"],
        "Statistiques": ["valid_predictions"]
    },
    1: {
        "Dépot et Validation des Données": ["valid_acceuil"],
        "Entraînement du Modèle": ["valid_depot_donnees"],
        "Prédictions": ["valid_entrainement"],
        "Statistiques": ["valid_predictions"]
    },
    2: {
        "Dépot du Modèle et des Données": ["valid_acceuil"],
        "Prédictions": ["valid_depot_donnees"],
        "Statistiques": ["valid_predictions"]
    }
}

# Dictionnaire pour les noms des pages en fonction de choix_modele
page_names = {
    0: {
        "valid_acceuil": "Accueil",
        "valid_depot_donnees": "Dépôt et Validation des Données",
        "valid_predictions": "Prédictions",
    },
    1: {
        "valid_acceuil": "Accueil",
        "valid_depot_donnees": "Dépôt et Validation des Données",
        "valid_entrainement": "Entraînement du Modèle",
        "valid_predictions": "Prédictions",
    },
    2: {
        "valid_acceuil": "Accueil",
        "valid_depot_donnees": "Dépôt du Modèle et des Données",
        "valid_predictions": "Prédictions",
    }
}

def check_dependencies(page_name):
    # Obtenir les dépendances en fonction de choix_modele
    dependencies = dependencies_by_choice.get(st.session_state.choix_modele, {}).get(page_name, [])
    for dependency in dependencies:
        if not st.session_state.get(dependency, False):
            # Utiliser le dictionnaire page_names pour obtenir le nom complet de la page
            page_display_name = page_names.get(st.session_state.choix_modele, {}).get(dependency, "Étape inconnue")
            st.error(f'Veuillez compléter l\'étape : "{page_display_name}" avant de continuer.')
            st.stop()
