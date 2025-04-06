
import plotly.io as pio
import zipfile
import io
from reportlab.pdfgen import canvas
import streamlit as st
import tempfile
from reportlab.lib.utils import ImageReader

def generer_csv(resultat_df, kpi_df, export_csv, export_prediction, export_kpi):
    """
    Génère les fichiers CSV à partir des DataFrames de prévision et de KPI, 
    selon les options d’exportation choisies par l’utilisateur.

    Paramètres :
        resultat_df (pd.DataFrame) : Données de prévision à exporter au format CSV.
        kpi_df (pd.DataFrame) : Données de métriques (KPI) à exporter au format CSV.
        export_csv (bool) : Indique si l'utilisateur a sélectionné le format CSV.
        export_prediction (bool) : Indique si les données de prévision doivent être incluses.
        export_kpi (bool) : Indique si les données de métriques doivent être incluses.

    Résultat retourné :
        - dict : Dictionnaire contenant :
            - les noms des fichiers CSV en tant que clés (ex. "export_predictions.csv"),
            - le contenu encodé en bytes comme valeurs.
            Si aucune exportation n’est activée, le dictionnaire est vide.
    """
    fichiers_csv = {}

    if export_csv:
        if export_prediction:
            fichiers_csv["export_predictions.csv"] = resultat_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        if export_kpi:
            fichiers_csv["export_metriques.csv"] = kpi_df.to_csv(index=False).encode('utf-8')

    return fichiers_csv


def generer_images(fig, tab):
    """
    Génère les images PNG à partir du graphique principal (`fig`) et du tableau des métriques (`tab`),
    en les convertissant en buffers mémoire utilisables pour les exports (PNG ou PDF).

    Paramètres :
        fig (plotly.graph_objects.Figure) : Graphique des prévisions à exporter.
        tab (plotly.graph_objects.Figure) : Tableau des métriques à exporter.

    Résultat retourné :
        - tuple :
            - fig_buffer (io.BytesIO) : Image PNG du graphique stockée en mémoire.
            - tab_buffer (io.BytesIO) : Image PNG du tableau stockée en mémoire.
    """
    # Mise à jour de la taille et des marges du graphique pour éviter qu'il soit tronqué
    fig.update_layout(width=1000, height=600, margin=dict(l=80, r=20, t=20, b=80))

    # Conversion du graphique en image PNG
    fig_bytes = pio.to_image(fig, format="png", engine="kaleido")
    #fig_bytes = pio.to_image(fig, format="png", engine="kaleido", scale=2)
    fig_buffer = io.BytesIO(fig_bytes)

    # Conversion du tableau en image PNG
    tab_bytes = pio.to_image(tab, format="png", engine="kaleido")
    tab_buffer = io.BytesIO(tab_bytes)

    return fig_buffer, tab_buffer


def generer_png(fig_buffer, tab_buffer, export_png, export_prediction, export_kpi):
    """
    Génère les fichiers PNG à partir des buffers mémoire contenant les images
    du graphique et du tableau, selon les options d’exportation sélectionnées.

    Paramètres :
        fig_buffer (io.BytesIO) : Image du graphique au format PNG (stockée en mémoire).
        tab_buffer (io.BytesIO) : Image du tableau des métriques au format PNG (stockée en mémoire).
        export_png (bool) : Indique si l'utilisateur a sélectionné l'export au format PNG.
        export_prediction (bool) : Indique si le graphique doit être inclus.
        export_kpi (bool) : Indique si le tableau des métriques doit être inclus.

    Résultat retourné :
        - dict : Dictionnaire contenant :
            - les noms des fichiers PNG comme clés ("export_graphique.png"),
            - le contenu des images encodé en bytes comme valeurs.
            Si aucune exportation n’est activée, le dictionnaire est vide.
    """
    
    fichiers_png = {}

    if export_png:
        if export_prediction:
            fichiers_png["export_graphique.png"] = fig_buffer.getvalue()
        if export_kpi:
            fichiers_png["export_metriques.png"] = tab_buffer.getvalue()

    return fichiers_png


def generer_pdf(fig_buffer, tab_buffer, export_pdf, export_prediction, export_kpi, titre_graphe):
    """
    Génère un rapport PDF contenant le graphique et le tableau des métriques, 
    selon les options d’exportation sélectionnées.

    Le fichier PDF est créé entièrement en mémoire, avec insertion des images (graphique + tableau),
    et gestion des erreurs d’insertion si nécessaire.

    Paramètres :
        fig_buffer (io.BytesIO) : Buffer contenant l’image PNG du graphique.
        tab_buffer (io.BytesIO) : Buffer contenant l’image PNG du tableau des métriques.
        export_pdf (bool) : Indique si l'utilisateur a sélectionné l'export au format PDF.
        export_prediction (bool) : Indique si le graphique doit être intégré au PDF.
        export_kpi (bool) : Indique si le tableau des métriques doit être intégré au PDF.
        titre_graphe (str) : Titre principal à afficher dans le rapport PDF.- plus utilisé

    Résultat retourné :
        - bytes : Contenu binaire du fichier PDF généré (ou None si `export_pdf` est désactivé).
    
    Exceptions gérées :
        - En cas d'erreur d'insertion des images, un message d'erreur est ajouté au PDF.
    """

    if not export_pdf:
        return None

    pdf_buffer = io.BytesIO()
    pdf_rapport = canvas.Canvas(pdf_buffer)

    # Titre principal
    pdf_rapport.setFont("Helvetica-Bold", size=14)
    pdf_rapport.setFillColorRGB(37/255, 41/255, 96/255)
    pdf_rapport.drawString(220, 800, "Export des résultats")
    y_position = 740

    # Insertion du graphique
    if export_prediction:
        pdf_rapport.setFont("Helvetica", 12)
        pdf_rapport.setFillColorRGB(24/255, 112/255, 184/255)
        pdf_rapport.drawString(100, y_position, "Graphique de prédiction des différents modèles obtenus")
        y_position -= 20

        try:
            fig_buffer.seek(0)
            pdf_rapport.drawImage(ImageReader(fig_buffer), 50, y_position - 235, width=500, height=250)
            y_position -= 280
        except Exception as e:
            pdf_rapport.setFont("Helvetica-Oblique", 10)
            pdf_rapport.setFillColorRGB(1, 0, 0)
            pdf_rapport.drawString(100, y_position - 20, "⚠️ Impossible de générer l’image du graphique.")
            pdf_rapport.drawString(100, y_position - 35, f"Erreur : {str(e)}")
            y_position -= 60

    # Insertion du tableau des métriques
    if export_kpi:
        y_position -= 20
        pdf_rapport.setFont("Helvetica", 12)
        pdf_rapport.setFillColorRGB(24/255, 112/255, 184/255)
        pdf_rapport.drawString(100, y_position, "Rappel des métriques")
        y_position -= 10

        try:
            tab_buffer.seek(0)
            pdf_rapport.drawImage(ImageReader(tab_buffer), 100, y_position - 200, width=400, height=200)
        except Exception as e:
            pdf_rapport.setFont("Helvetica-Oblique", 10)
            pdf_rapport.setFillColorRGB(1, 0, 0)
            pdf_rapport.drawString(100, y_position - 20, "⚠️ Impossible de générer l’image du tableau.")
            pdf_rapport.drawString(100, y_position - 35, f"Erreur : {str(e)}")

    pdf_rapport.showPage()
    pdf_rapport.save()
    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()

def creer_zip(fichiers_a_exporter, zip_file_path):
    """
    Crée un fichier ZIP contenant les fichiers à exporter (CSV, PNG, PDF, etc.),
    à partir d’un dictionnaire {nom_fichier: contenu}.

    Les fichiers sont compressés et stockés à l’emplacement zip_file_path.

    Paramètres :
        fichiers_a_exporter (dict) : Dictionnaire contenant :
            - les noms de fichiers comme clés (str),
            - leur contenu binaire (bytes) comme valeurs.
        zip_file_path (str) : Chemin complet du fichier ZIP à créer.

    Résultat retourné :
        - str : Chemin du fichier ZIP généré.
    """
    with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename, content in fichiers_a_exporter.items():
            zip_file.writestr(filename, content)

    return zip_file_path


def export_data_zip(resultat_df, kpi_df, export_options, fig, tab, titre_graphe, zip_file_path):
    """
    Exporte l’ensemble des éléments sélectionnés (données, graphique, tableau) 
    dans un fichier ZIP, selon les options définies par l’utilisateur.

    Cette fonction orchestre l’ensemble du processus en fonction des sélections:
    - Génération des fichiers  CSV des prédictions et métrique,
    - Génération des images PNG du graphique et du tableau,
    - Création d’un rapport PDF contenant les visuels,
    - Compression de tous les fichiers dans un ZIP.

    Paramètres :
        resultat_df (pd.DataFrame) : Données de prévision à exporter.
        kpi_df (pd.DataFrame) : Données de métriques (KPI) à exporter.
        export_options (dict) : Dictionnaire contenant les options d’export sélectionnées :
            - export_options["formats"]["export_format_csv"] : bool
            - export_options["formats"]["export_format_png"] : bool
            - export_options["formats"]["export_format_pdf"] : bool
            - export_options["donnees"]["export_prediction"] : bool
            - export_options["donnees"]["export_kpi"] : bool
        fig (plotly.graph_objects.Figure) : Graphique à inclure dans le PNG et PDF.
        tab (plotly.graph_objects.Figure) : Tableau des métriques à inclure dans le PNG et PDF.
        titre_graphe (str) : Titre affiché dans le rapport PDF.
        zip_file_path (str) : Chemin du fichier ZIP à créer.

    Résultat retourné :
        - str : Chemin du fichier ZIP généré contenant tous les fichiers sélectionnés.
    """
    # Stockage des options dans variables locales
    export_csv = export_options["formats"]["export_format_csv"]
    export_png = export_options["formats"]["export_format_png"]
    export_pdf = export_options["formats"]["export_format_pdf"]
    
    export_prediction = export_options["donnees"]["export_prediction"]
    export_kpi = export_options["donnees"]["export_kpi"]

    files_to_zip = {}

    # Génération des fichiers CSV
    files_to_zip.update(
        generer_csv(resultat_df, kpi_df, export_csv, export_prediction, export_kpi)
    )

    #  Génération des images (graphique + tableau)
    fig_buffer, tab_buffer = generer_images(fig, tab)

    #  Génération des fichiers PNG
    files_to_zip.update(
        generer_png(fig_buffer, tab_buffer, export_png, export_prediction, export_kpi)
    )

    # Génération du fichier PDF
    contenu_pdf = generer_pdf(fig_buffer, tab_buffer, export_pdf, export_prediction, export_kpi, titre_graphe)
    if contenu_pdf:
        files_to_zip["rapport_donnees_selectionnees.pdf"] = contenu_pdf

    # Création du ZIP
    chemin_fichier_zip=creer_zip(files_to_zip, zip_file_path)
    return chemin_fichier_zip

def gerer_export(df_final_selection, df_kpi_selection, export_options, fig, tab, titre_graphe):
   
    """
    Gère le processus global d'exportation en ZIP à partir des données filtrées 
    et des options sélectionnées par l'utilisateur via l'interface Streamlit.

    Cette fonction :
    - Crée un fichier ZIP temporaire contenant les fichiers souhaités (CSV, PNG, PDF),
    - Met à jour l'état de Streamlit (`st.session_state`) pour indiquer que l'export est prêt.

    Paramètres :
        df_final_selection (pd.DataFrame) : Données filtrées à exporter (prévisions).
        df_kpi_selection (pd.DataFrame) : Métriques KPI filtrées à exporter.
        export_options (dict) : Dictionnaire contenant les options d’exportation choisies.
        fig (plotly.graph_objects.Figure) : Graphique à intégrer dans les exports.
        tab (plotly.graph_objects.Figure) : Tableau des métriques à intégrer dans les exports.
        titre_graphe (str) : Titre à afficher dans le rapport PDF.

    Effets secondaires :
        - Met à jour :
            - `st.session_state.zip_ready` : booléen indiquant si le ZIP est prêt.
            - `st.session_state.zip_path` : chemin du fichier ZIP généré.
    """
    st.session_state.zip_ready = False
    with st.spinner("⏳ Préparation du fichier ZIP en cours... Veuillez patienter."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            st.session_state.zip_path = export_data_zip(df_final_selection, df_kpi_selection,
                                                        export_options, fig, tab, titre_graphe,tmp_zip.name)
        st.session_state.zip_ready = True
