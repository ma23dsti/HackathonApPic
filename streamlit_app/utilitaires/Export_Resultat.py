
#from fpdf import FPDF
import plotly.io as pio
import zipfile
import io
from reportlab.pdfgen import canvas
import streamlit as st
import tempfile
import os


#fonction pour exporter les données en csv, pdf apres les avoir zippé
#def  export_data_zip(resultat_df, kpi_df, choix_donnees_export, choix_format_export,donnees_prevision, donnees_kpi,fig, tab, titre_graphe,dpi_value):   
def  export_data_zip(resultat_df, kpi_df, choix_donnees_export, choix_format_export,donnees_prevision, donnees_kpi,fig, tab, titre_graphe):   

    #export_file = {}  # Stocke les fichiers exportés
    zip_buffer = io.BytesIO()  # Création d'un fichier ZIP en mémoire

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:

        # Export CSV
        if "CSV" in choix_format_export:
            # récuperation des données liés aux prévisions et données d'entrée
            if donnees_prevision in choix_donnees_export:
                csv_data = resultat_df.to_csv(index=False).encode('utf-8')
                zip_file.writestr("export_previsions.csv", csv_data)  # Ajout dans le ZIP
            # récuperation des métriques
            if donnees_kpi in choix_donnees_export:
                csv_data = kpi_df.to_csv(index=False).encode('utf-8')
                zip_file.writestr("export_metriques.csv", csv_data)


        # Export PNG
        if "PNG" in choix_format_export:
            dpi = 150  # Valeur par défaut en attendant selection
            png_buffer = io.BytesIO()
            fig.savefig(png_buffer, format="png", dpi=dpi, bbox_inches="tight")  # Sauvegarde  de l'image
            png_buffer.seek(0)
            zip_file.writestr("export_graph.png", png_buffer.read())

        # Export PDF
        if "PDF" in choix_format_export:

            # Récupérer le dossier temporaire du système
            temp_folder = tempfile.gettempdir()

            pdf_buffer=io.BytesIO()
            pdf_rapport=canvas.Canvas(pdf_buffer)

            #Titre du rapport
            pdf_rapport.setFont("Helvetica-Bold", size=14) # Définition de la police
            pdf_rapport.setFillColorRGB(0, 0, 1)  # Bleu
            pdf_rapport.drawString(220, 800, "Export des résultats")

            y_position=740 #ajout espace apres le titre

            # Affichage du graphique des prévisions
            if donnees_prevision in choix_donnees_export:
                pdf_rapport.setFont("Helvetica", 12)
                pdf_rapport.setFillColorRGB(0, 0, 1)  # Bleu
                pdf_rapport.drawString(100, y_position, titre_graphe)
                
                y_position -= 20 # réduction espace avant le graphe



                # Générer un fichier temporaire 
                temp_graph_path = os.path.join(temp_folder, "temp_graph.png")

                # Sauvegarde du graphique
                fig.savefig(temp_graph_path, format="png")

                # Utilisation du fichier dans le PDF
                pdf_rapport.drawImage(temp_graph_path, 50,  y_position - 235, width=500, height=250)

                y_position -= 280 # ajustement espace entre graphe et tableau

                # Suppression du fichier après utilisation
                os.remove(temp_graph_path)

            # Affichage du tableau des métriques
            if donnees_kpi in choix_donnees_export:
                y_position-=20
                pdf_rapport.setFont("Helvetica", 12)
                pdf_rapport.setFillColorRGB(0, 0, 1)  # Bleu
                pdf_rapport.drawString(100, y_position, "Rappel des métriques")
                
                y_position-=10

                # Générer un fichier temporaire 
                temp_tab_path = os.path.join(temp_folder, "temp_tab.png")

                # Sauvegarde du tableau
                fig.savefig(temp_tab_path, format="png")

                # Utilisation du fichier dans le PDF
                pio.write_image(tab, temp_tab_path)
                pdf_rapport.drawImage(temp_tab_path, 100, y_position-200, width=400, height=200)

                # Suppression du fichier après utilisation
                try:
                    os.remove(temp_tab_path)
                except Exception as e:
                    st.write(f"Erreur lors de la suppression de {temp_tab_path}: {e}")

            pdf_rapport.showPage()
            pdf_rapport.save()

            # Remettre le buffer au début et ajouter au ZIP
            pdf_buffer.seek(0)
            zip_file.writestr("rapport_donnees_selectionnes2.pdf", pdf_buffer.read())

    zip_buffer.seek(0)  # Remettre le fichier ZIP en lecture
    return zip_buffer.getvalue()

