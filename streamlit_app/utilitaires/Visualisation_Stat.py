import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# Définition des palettes
palette_defaut = px.colors.qualitative.Plotly 


# Définition des styles de lignes et des épaisseurs (12 styles différents)
ligne_styles = [':','-', '--', '-.']  # Plein, tirets, mixte, pointillé
ligne_epaisseur = [1, 1.5, 2]  

#Fonction de creation du graphe des prévisions

def creation_graphique(df_data,palette,id_modele,var_id,var_x,var_y,label_x,label_y):
   
    # Création de la figure
    fig, ax = plt.subplots(figsize=(10, 5))
    # boucle sur les id_modele pour récuperer les données et créer le tableau
    for i, modele in enumerate(id_modele):
        df_modele=df_data[df_data[var_id]==modele]
        nom_modele = df_modele["nom donnee"].iloc[0] # Récupération du nom du modèle
        ax.plot(df_modele[var_x], df_modele[var_y], 
                label=nom_modele,
                color=palette[i % len(palette)], 
                linestyle=ligne_styles[i % len(ligne_styles)], 
                linewidth=ligne_epaisseur[i % len(ligne_epaisseur)])
    

    # Ajout des titres et labels
    ax.set_xlabel(label_x, fontsize=12)
    ax.set_ylabel(label_y, fontsize=12)

    # Ajout de la légende
    ax.legend()
    plt.tight_layout()

    return fig
  


#Fonction pour créer le tableau des métriques
def creation_tableau (df_kpi_selection):

    # Pivot pour avoir les indicateurs en colonnes 
    df_kpi_pivot = df_kpi_selection.pivot(index='nom donnee', columns='indicateur', values='valeur').reset_index()

    # Création du tableau 
    tab = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>{}</b>'.format(col.upper()) for col in df_kpi_pivot.columns],  # Mise en gras et majuscule des titres
            fill_color='#e86e4c',  # Couleur de fond des en-têtes
            font=dict(color='black', size=12),  # Couleur et taille du texte des en-têtes
            align='left',  # Alignement du texte
            height=30,  # Hauteur des en-têtes
            line=dict(color='black', width=2)  # Couleur et épaisseur des traits
        ),
        cells=dict(
            values=[df_kpi_pivot[col] for col in df_kpi_pivot.columns],  # Remplissage des cellules
            fill_color='#fbb458',  # Couleur de fond du corps du tableau
            font=dict(color='black', size=12),  # Couleur et taille du texte des cellules
            align='left',  # Alignement du texte
            height=25,  # Hauteur des cellules
            line=dict(color='black', width=2)  # Couleur et épaisseur des traits
        )
    )])
    tab.update_layout(
        autosize=False,  # Désactive l'ajustement automatique
        height=350,
        margin=dict(t=0, l=0, r=0, b=0)  # Suppression totale des marges pour éviter tout centrage vertical
    )
 
    return tab 


    

