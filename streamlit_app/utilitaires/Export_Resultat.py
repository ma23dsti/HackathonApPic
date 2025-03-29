
#from fpdf import FPDF
import plotly.io as pio
import zipfile
import io
from reportlab.pdfgen import canvas
import streamlit as st
import tempfile
import os
from reportlab.lib.utils import ImageReader



#fonction pour créé un Zip des données sélectionées selon formats sélectionnés d'export

def export_data_zip(resultat_df, kpi_df, export_options, fig, tab, titre_graphe, zip_file_path):
    # Stockage des options dans variables locales
    export_csv = export_options["formats"]["export_format_csv"]
    export_png = export_options["formats"]["export_format_png"]
    export_pdf = export_options["formats"]["export_format_pdf"]
    
    export_prevision = export_options["donnees"]["export_prevision"]
    export_kpi = export_options["donnees"]["export_kpi"]

    files_to_zip = {}  # Stocke les fichiers avant de les écrire dans le ZIP

    # Export CSV
    if export_csv:
        if export_prevision:
            files_to_zip["export_previsions.csv"] = resultat_df.to_csv(index=False).encode('utf-8')

        if export_kpi:
            files_to_zip["export_metriques.csv"] = kpi_df.to_csv(index=False).encode('utf-8')

    #  Génération des images et stockage en mémoire
    img_buffer = io.BytesIO()
    #fig.write_image(img_buffer, format="png")
    fig.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
    img_buffer.seek(0)  # Remet le buffer au début pour lecture

    # Export PNG
    if export_png:
        files_to_zip["export_graph.png"] = img_buffer.getvalue()  # Utilise directement le buffer en mémoire

    # Export PDF
    if export_pdf:
        pdf_buffer = io.BytesIO()
        pdf_rapport = canvas.Canvas(pdf_buffer)

        # Titre du rapport
        pdf_rapport.setFont("Helvetica-Bold", size=14)
        pdf_rapport.setFillColorRGB(0, 0, 1)
        pdf_rapport.drawString(220, 800, "Export des résultats")

        y_position = 740  # Espace après le titre

        # Ajout du graphique
        if export_prevision:
            pdf_rapport.setFont("Helvetica", 12)
            pdf_rapport.setFillColorRGB(0, 0, 1)
            pdf_rapport.drawString(100, y_position, titre_graphe)
            y_position -= 20

            # Utilisation du buffer existant pour éviter une nouvelle sauvegarde
            pdf_rapport.drawImage(ImageReader(img_buffer), 50, y_position - 235, width=500, height=250)
            y_position -= 280

        # Ajout du tableau des métriques
        if export_kpi:
            y_position -= 20
            pdf_rapport.setFont("Helvetica", 12)
            pdf_rapport.setFillColorRGB(0, 0, 1)
            pdf_rapport.drawString(100, y_position, "Rappel des métriques")
            y_position -= 10

            # Génération de l'image du tableau
            tab_bytes = pio.to_image(tab, format="png", engine="kaleido")
            tab_buffer = io.BytesIO(tab_bytes)
            #tab_buffer = io.BytesIO()
            #pio.write_image(tab, tab_buffer, format="png")
            #tab_buffer.seek(0)

            pdf_rapport.drawImage(ImageReader(tab_buffer), 100, y_position - 200, width=400, height=200)

        pdf_rapport.showPage()
        pdf_rapport.save()

        pdf_buffer.seek(0)
        files_to_zip["rapport_donnees_selectionnees.pdf"] = pdf_buffer.getvalue()

    # Écriture du ZIP
    with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename, content in files_to_zip.items():
            zip_file.writestr(filename, content)

    return zip_file_path


