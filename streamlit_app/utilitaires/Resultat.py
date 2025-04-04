from datetime import datetime, timedelta
import json
import os
import random
import streamlit as st

def mettre_a_jour_model_info(predictions, entrainement_modele=False):
    """Met à jour model_info avec la nouvelle prédiction et recalcule modele_moyen."""

    if 'model_info' not in st.session_state:
        st.session_state.model_info = []
    model_info = st.session_state.model_info
    # Génération d'un ID et nom de modèle aléatoires
    model_id = len([m for m in model_info if m["id"] != "moyenne"]) + 1
    model_name = f"model_{model_id}"

    # TO DO Génération de MAE aléatoires
    # Ce code pourrait être factorisé comme il est aussi dans celui de la page Prédictions
    if 'entrainement_modele' in st.session_state:
        if st.session_state.entrainement_modele==True:
            nrmse_courant = st.session_state.nrmse_value
            mae_courant = st.session_state.mae_value
    else: 
        with open("streamlit_app/static/modeles/modele_par_defaut/modele_parametres.json", "r") as f:
            params = json.load(f)
            nrmse_courant = params["kpi"]["nrmse"]
            mae_courant = round(random.uniform(0.3, 0.5), 4)

    kpi_values = {"mae": mae_courant, "nrmse": nrmse_courant}
    
    # Ajouter la nouvelle prédiction à l'historique
    model_entry = {
        "id": model_id,
        "nom": model_name,
        "kpi": kpi_values,
        "donnees_predites": predictions.tolist(),
    }
    model_info.append(model_entry)

    # Recalculer modele_moyen
    models_without_average = [m for m in model_info if m["id"] != "moyenne"]
    
    if models_without_average:
        avg_kpi = {
            "mae": round(sum(m["kpi"]["mae"] for m in models_without_average) / len(models_without_average), 4),
            "nrmse": round(sum(m["kpi"]["nrmse"] for m in models_without_average) / len(models_without_average), 4),
        }
        avg_predictions = [sum(p) / len(models_without_average) for p in zip(*[m["donnees_predites"] for m in models_without_average])]
    else:
        avg_kpi = {"mae": 0, "nrmse": 0}
        avg_predictions = []

    modele_moyen = {
        "id": "moyenne",
        "nom": "modele_moyen",
        "kpi": avg_kpi,
        "donnees_predites": avg_predictions,
    }

    # Supprimer l'ancien modele_moyen et ajouter le nouveau
    model_info = [m for m in model_info if m["id"] != "moyenne"] + [modele_moyen]

    st.session_state.model_info = model_info

    # Determine unite_mesure_2 based on unite_mesure
    unite_mesure = st.session_state.unite_mesure
    if unite_mesure == "Bits/s":
        unite_mesure_2 = "Octets/s"
        conversion_factor = 1 / 8  # Convert Bits to Octets
    elif unite_mesure == "Octets/s":
        unite_mesure_2 = "Bits/s"
        conversion_factor = 8  # Convert Octets to Bits
    else:
        raise ValueError("Unité de mesure inconnue. Les valeurs acceptées sont 'Bits/s' ou 'Octets/s'.")

    # Convert donnees_observees to both units
    donnees_observees = st.session_state.prediction_data.values.flatten().tolist()  # Ensure it's a NumPy array before flattening
    donnees_observees_converted = [round(value * conversion_factor, 2) for value in donnees_observees]

    # Convert donnees_predites to both units
    predictions_converted = [round(value * conversion_factor, 2) for value in predictions.tolist()]

    # Generate the "temps_horaire" list
    date_premiere_observation = datetime.strptime(st.session_state.date_premiere_observation, "%Y-%m-%d %H:%M:%S")
    temps_horaire = [
        (date_premiere_observation + timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(st.session_state.taille_fenetre_observee + st.session_state.horizon_predictions)
    ]

    # Construct the JSON object
    resultats = {
        "donnees_entrees": {
            "nom_fichier": "donnees_traffic.csv",
            "periodicite_mesure": "s",
            "unite_mesure": unite_mesure,
            "unite_mesure_2": unite_mesure_2,
            "format_mesure": "",
            "temps_horaire": temps_horaire[:st.session_state.taille_fenetre_observee],
            "temps_relatif": list(range(1, st.session_state.taille_fenetre_observee + 1)),
            "donnees_observees": {
                "en_unite_mesure": donnees_observees,
                "en_unite_mesure_2": donnees_observees_converted
            }
        },
        "resultats": {
            "predictions": {
                "temps_relatif": list(range(st.session_state.taille_fenetre_observee + 1, st.session_state.taille_fenetre_observee + 1 + st.session_state.horizon_predictions)),
                "temps_horaire": temps_horaire[st.session_state.taille_fenetre_observee:],
                "modeles": [
                    {
                        **modele,
                        "donnees_predites": {
                            "en_unite_mesure": modele["donnees_predites"],
                            "en_unite_mesure_2": [round(value * conversion_factor, 2) for value in modele["donnees_predites"]]
                        }
                    }
                    for modele in st.session_state.model_info
                ]
            }
        }
    }

    st.session_state.resultats = resultats
    resultats_nom_fichier = "resultats.json"

    resultats_chemin_fichier = os.path.join("streamlit_app/resultats/donnees_a_la_volee", resultats_nom_fichier)
    os.makedirs(os.path.dirname(resultats_chemin_fichier), exist_ok=True)
    with open(resultats_chemin_fichier, "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=4)

    return resultats