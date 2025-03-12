from datetime import datetime
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
    else: 
        with open("streamlit_app/static/modeles/modele_par_defaut/modele_parametres.json", "r") as f:
            params = json.load(f)
            nrmse_courant = params["kpi"]["nrmse"]
    kpi_values = {"mae": round(random.uniform(0.3, 0.5), 4), "nrmse": nrmse_courant}
    
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

    # Construire le JSON complet
    resultats = {
        "donnees_entrees": {
        "nom_fichier": "donnees_traffic.csv",
        "periodicite_mesure": "s",
        "unite_mesure": "Bits/s",
        "format_mesure": "",
        "temps_horaire": ["2025-02-10 00:01:00", "2025-02-10 00:01:01", "2025-02-10 00:01:02", "2025-02-10 00:01:03", "2025-02-10 00:01:04", 
      "2025-02-10 00:01:05", "2025-02-10 00:01:06", "2025-02-10 00:01:07", "2025-02-10 00:01:08", "2025-02-10 00:01:09", 
      "2025-02-10 00:01:10", "2025-02-10 00:01:11", "2025-02-10 00:01:12", "2025-02-10 00:01:13", "2025-02-10 00:01:14", 
      "2025-02-10 00:01:15", "2025-02-10 00:01:16", "2025-02-10 00:01:17", "2025-02-10 00:01:18", "2025-02-10 00:01:19", 
      "2025-02-10 00:01:20", "2025-02-10 00:01:21", "2025-02-10 00:01:22", "2025-02-10 00:01:23", "2025-02-10 00:01:24", 
      "2025-02-10 00:01:25", "2025-02-10 00:01:26", "2025-02-10 00:01:27", "2025-02-10 00:01:28", "2025-02-10 00:01:29", 
      "2025-02-10 00:01:30", "2025-02-10 00:01:31", "2025-02-10 00:01:32", "2025-02-10 00:01:33", "2025-02-10 00:01:34", 
      "2025-02-10 00:01:35", "2025-02-10 00:01:36", "2025-02-10 00:01:37", "2025-02-10 00:01:38", "2025-02-10 00:01:39", 
      "2025-02-10 00:01:40", "2025-02-10 00:01:41", "2025-02-10 00:01:42", "2025-02-10 00:01:43", "2025-02-10 00:01:44", 
      "2025-02-10 00:01:45", "2025-02-10 00:01:46", "2025-02-10 00:01:47", "2025-02-10 00:01:48", "2025-02-10 00:01:49", 
      "2025-02-10 00:01:50", "2025-02-10 00:01:51", "2025-02-10 00:01:52", "2025-02-10 00:01:53", "2025-02-10 00:01:54", 
      "2025-02-10 00:01:55", "2025-02-10 00:01:56", "2025-02-10 00:01:57", "2025-02-10 00:01:58", "2025-02-10 00:01:59"],
        "temps_relatif": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                           11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                           21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                           31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                           41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                           51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
        "donnees_observees": st.session_state.prediction_data.values.flatten().tolist()
        #"donnees_observees": [228664,11624,20584,5648,71096,7932,82552,24512,129808,61712,81792,16748,344048,381085,2086,125152,7976,76976,97608,21248,1628,33096,3592,19104,62936,13328,86728,119872,45504,21936,20864,11176,21528,284,20352,8312,16176,57672,24144,33616,75696,2844,1796,3268,10784,31192,18752,115392,48296,39576,8476,32568,27256,38408,31152,252,23448,1544,3276,41584]
        },
        "resultats": {
            "predictions": {
                "temps_relatif": list(range(61, 66)),  # Temps relatif
                "temps_horaire": [
                    "2025-02-10 00:02:00",
                    "2025-02-10 00:02:01",
                    "2025-02-10 00:02:02",
                    "2025-02-10 00:02:03",
                    "2025-02-10 00:02:04"
                ],
                "modeles": model_info
            }
        }
    }

    st.session_state.resultats = resultats

    resultats_nom_fichier = "resultats.json"
    #resultats_nom_fichier = f"resultats_{modele_moyen["kpi"]["nrmse"]:.4f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    resultats_chemin_fichier = os.path.join("streamlit_app/resultats/donnees_a_la_volee", resultats_nom_fichier)
    os.makedirs(os.path.dirname(resultats_chemin_fichier), exist_ok=True)
    with open(resultats_chemin_fichier, "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=4)
    st.write("Fichier des résultats sauvegardé : ", resultats_nom_fichier)

    return resultats