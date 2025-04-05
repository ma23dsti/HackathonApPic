
import plotly.io as pio
import zipfile
import io
from reportlab.pdfgen import canvas
import streamlit as st
import tempfile
from reportlab.lib.utils import ImageReader

def generer_csv(resultat_df, kpi_df, export_csv, export_prediction, export_kpi):
    """
    Génère les fichiers CSV à partir des DataFrames fournis selon les options d'export.

    Paramètres :
    ----------
    resultat_df : pd.DataFrame
        Données de prévision à exporter.
    kpi_df : pd.DataFrame
        Données de métriques à exporter.
    export_csv : bool
        Indique si le format CSV est sélectionné.
    export_prediction : bool
        Indique si les données de prévision doivent être exportées.
    export_kpi : bool
        Indique si les données de KPI doivent être exportées.

    Retour :
    -------
    dict :
        Dictionnaire avec les noms de fichiers CSV comme clés et le contenu encodé en bytes comme valeurs.
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
    Génère les images PNG du graphique principal et du tableau des métriques,
    et les stocke dans des buffers mémoire.

    Paramètres :
    ----------
    fig : plotly.graph_objects.Figure
        Graphique à exporter.
    tab : plotly.graph_objects.Figure
        Tableau des métriques à exporter.

    Retour :
    -------
    tuple :
        - fig_buffer : io.BytesIO contenant l’image PNG du graphique.
        - tab_buffer : io.BytesIO contenant l’image PNG du tableau.
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
    Génère les fichiers PNG à partir des buffers d’images du graphique et du tableau.

    Paramètres :
    ----------
    fig_buffer : io.BytesIO
        Buffer contenant l’image PNG du graphique.
    tab_buffer : io.BytesIO
        Buffer contenant l’image PNG du tableau des métriques.
    export_png : bool
        Indique si le format PNG est sélectionné.
    export_prediction : bool
        Indique si les données de prévision doivent être exportées.
    export_kpi : bool
        Indique si les données de KPI doivent être exportées.

    Retour :
    -------
    dict :
        Dictionnaire avec les noms de fichiers PNG comme clés et leur contenu en bytes comme valeurs.
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
    Crée un rapport PDF contenant le graphique et le tableau des métriques selon les options sélectionnées.

    Paramètres :
    ----------
    fig_buffer : io.BytesIO
        Buffer contenant l’image PNG du graphique.
    tab_buffer : io.BytesIO
        Buffer contenant l’image PNG du tableau.
    export_pdf : bool
        Indique si le format PDF est sélectionné.
    export_prediction : bool
        Indique si le graphique doit être inclus dans le PDF.
    export_kpi : bool
        Indique si le tableau des métriques doit être inclus dans le PDF.
    titre_graphe : str
        Titre du graphique à afficher dans le rapport.

    Retour :
    -------
    bytes :
        Contenu du fichier PDF en mémoire (ou None si non généré).
    """
    if not export_pdf:
        return None

    pdf_buffer = io.BytesIO()
    pdf_rapport = canvas.Canvas(pdf_buffer)

    # Titre principal
    pdf_rapport.setFont("Helvetica-Bold", size=14)
    pdf_rapport.setFillColorRGB(0, 0, 1)
    pdf_rapport.drawString(220, 800, "Export des résultats")
    y_position = 740

    # Insertion du graphique
    if export_prediction:
        pdf_rapport.setFont("Helvetica", 12)
        pdf_rapport.setFillColorRGB(0, 0, 1)
        pdf_rapport.drawString(100, y_position, titre_graphe)
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
        pdf_rapport.setFillColorRGB(0, 0, 1)
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
    Crée un fichier ZIP contenant tous les fichiers spécifiés.

    Paramètres :
    ----------
    fichiers_a_exporter : dict
        Dictionnaire avec noms de fichiers comme clés et contenu en bytes comme valeurs.
    zip_file_path : str
        Chemin complet où le fichier ZIP sera créé.

    Retour :
    -------
    str :
        Chemin du fichier ZIP généré.
    """
    with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename, content in fichiers_a_exporter.items():
            zip_file.writestr(filename, content)

    return zip_file_path


def export_data_zip(resultat_df, kpi_df, export_options, fig, tab, titre_graphe, zip_file_path):
    """
    Exporte les données sélectionnées (prévisions, KPI, graphique, tableau) dans un fichier ZIP.

    En fonction des options cochées par l'utilisateur (`export_options`), cette fonction :
    - Génère les fichiers CSV des données de prévision et des métriques (KPI).
    - Convertit le graphique (`fig`) en image PNG.
    - Crée un rapport PDF contenant le graphique et le tableau des métriques (`tab`).
    - Regroupe tous les fichiers dans un fichier ZIP temporaire à l'emplacement spécifié (`zip_file_path`).

    Paramètres :
    ----------
    resultat_df : pd.DataFrame
        Données de prévision à exporter.
    kpi_df : pd.DataFrame
        Données de métriques à exporter.
    export_options : dict
        Dictionnaire contenant les choix de formats et types de données à exporter.
    fig : matplotlib.figure.Figure ou plotly.graph_objects.Figure
        Graphique à exporter.
    tab : plotly.graph_objects.Figure
        Tableau des métriques à inclure dans le PDF.
    titre_graphe : str
        Titre à afficher dans le rapport PDF.
    zip_file_path : str
        Chemin complet où le fichier ZIP sera temporairement enregistré.

    Retour :
    -------
    str :
        Le chemin du fichier ZIP généré.
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
    Gère la génération du fichier ZIP en fonction des options sélectionnées.
    Appelle la fonction export_data_zip pour créer le fichier zip temporaire
    Met à jour st.session_state['zip_ready'] et ['zip_path'].
    """
    st.session_state.zip_ready = False
    with st.spinner("⏳ Préparation du fichier ZIP en cours... Veuillez patienter."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            st.session_state.zip_path = export_data_zip(df_final_selection, df_kpi_selection,
                                                        export_options, fig, tab, titre_graphe,tmp_zip.name)
        st.session_state.zip_ready = True
