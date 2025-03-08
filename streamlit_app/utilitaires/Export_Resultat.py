
from fpdf import FPDF
import plotly.io as pio
import zipfile
import io





#fonction pour exporter les données en csv, pdf apres les avoir zippé
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

        # Export PDF
        if "PDF" in choix_format_export:
            pdf = FPDF()
            pdf.add_page()

            temp_graph_path = "temp_graph.png" # graphiques des prédictions et données d'entrée
            temp_tab_path = "temp_tab.png" # tableau des métriques


            pdf.set_font("Arial", style='B', size=14) # Définition de la police
            pdf.set_text_color(0, 0, 255)  # Bleu
            pdf.cell(200, 10, "Export des résultats", ln=True, align='C')
            
            # affichage du graphique des prévisions
            if donnees_prevision in choix_donnees_export: 
                # Ajout titre avant graphe
                pdf.ln(10)  # Ajoute un espace avant le titre
                pdf.set_font("Arial", style='', size=12) 
                pdf.set_text_color(0, 0, 255)  # Bleu
                pdf.cell(200, 10, f"{titre_graphe}", ln=True, align='L')  #titre récupéré par la sélection des paramétrespdf.ln(5)
                pdf.ln(8)
        

                # Sauvegarde du graphique en fichier temporaire
                fig.savefig(temp_graph_path, format="png")
                # Ajout du graphique dans le PDF après l'enregistrement
                pdf.image(temp_graph_path, x=10, y=pdf.get_y() -10, w=180)
            
            # affichage du tableau des métriques
            if donnees_kpi in choix_donnees_export:
                pdf.ln(85)
                pdf.cell(200, 10, "Rappel des métriques", ln=True, align='L')
                pdf.ln(105)

                # Sauvegarde du tableau en fichier temporaire
                pio.write_image(tab, temp_tab_path)  
                # Ajout l’image du tableau au PDF
                pdf.image(temp_tab_path, x=10, y=pdf.get_y()-100, w=180)

            pdf_buffer = io.BytesIO()
            pdf.output(pdf_buffer)
            pdf_buffer.seek(0)
            zip_file.writestr("rapport_donnees_selectionnes2.pdf", pdf_buffer.read())  # Ajout dans le ZIP

    zip_buffer.seek(0)  # Remettre le fichier ZIP en lecture

    return zip_buffer.getvalue()

