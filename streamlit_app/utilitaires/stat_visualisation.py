import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# Définition des palettes
palette_defaut = px.colors.qualitative.Plotly 

palette_daltonien=[
                    "rgb(152,202,225)",
                    "rgb(37,42,184)",
                     "rgb(204,76,2)",
                    "rgb(102,37,6)",
                    "rgb(74,123,183)",
                   "rgb(153,52,4)",
                  "rgb(236,112,20)",
                     "rgb(110,166,205)",
                      "rgb(240,173,50)",
                  "rgb(251,154,41)",
                 "rgb(57,75,154)",
                    "rgb(230,85,13)"
                  ]


# Définition des styles de lignes et des épaisseurs

ligne_styles =['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
ligne_epaisseur = [2, 3, 3.5]  


def creation_graphique(df_data, palette, id_modele, var_id, var_x, var_y, label_x, label_y):
    """
    Crée un graphique interactif avec Plotly affichant les courbes des données sélectionnées
    (données d'entrée et/ou prédictions) selon les modèles fournis.

    Chaque courbe est personnalisée en couleur, style et épaisseur de ligne, avec une légende 
    et un survol informatif.
    Si aucune donnée n'est trouvée, un message est affiché à la place du graphe.

    Paramètres :
        df_data (pd.DataFrame) : Données filtrées à afficher (entrées ou prédictions).
        palette (list[str]) : Liste des couleurs utilisées pour différencier les modèles.
        id_modele (list[str]) : Liste des identifiants des modèles à afficher.
        var_id (str) : Nom de la colonne identifiant les séries (ex. 'id donnee').
        var_x (str) : Nom de la colonne à utiliser pour l’axe des X (ex. 'temps relatif').
        var_y (str) : Nom de la colonne à utiliser pour l’axe des Y (ex. 'valeur').
        label_x (str) : Étiquette de l’axe X.
        label_y (str) : Étiquette de l’axe Y.

    Résultat retourné :
        - go.Figure : Graphique Plotly avec les courbes des modèles sélectionnés 
        ou un message d’absence de données.
    """
    # Création de la figure Plotly
    fig = go.Figure()

    modeles_trouves= False # check si au moins une courbe peut être créée

    # Boucle sur les modèles pour récupérer les données et tracer chaque courbe
    for i, modele in enumerate(id_modele):
        df_modele = df_data[df_data[var_id] == modele] # Filtrage des données pour le modèle courant
        nom_modele = df_modele["nom donnee"].iloc[0] # Récupération du nom du modèle
        donnees_trouvees=True # on peut tracer au moins une courbe

        # Ajout de la courbe au graphique
        fig.add_trace(go.Scatter(
            x=df_modele[var_x],
            y=df_modele[var_y], 
           # text=None,
            #hovertemplate='%{fullData.name}<br>x: %{x}<br>y: %{y}<extra></extra>',
            mode='lines+markers' if len(df_modele) == 1 else 'lines', # Type de courbe et point si 
            name=nom_modele, # Légende
            line=dict(
                color=palette[i % len(palette)], # Couleur personnalisée
                dash=ligne_styles[i % len(ligne_styles)], # Style de ligne
                width=ligne_epaisseur[i % len(ligne_epaisseur)] # Épaisseur de ligne

            ),
            showlegend=True
        ))

        # Si aucune donnée n'a été sélectionnée, affiche un message
        if not donnees_trouvees:
            fig.add_annotation(
                text="Aucune donnée n'a été sélectionnée dans les paramétres",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16),
                xanchor='center',
                yanchor='middle'
            )
            # Masque les axes du graphe
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )    

    # Mise à jour des axes et options d’interaction
    fig.update_layout(
        legend=dict(font=dict(color='black'), bordercolor='black', borderwidth=1, orientation="h",
                    yanchor="middle", y=1.18,xanchor="center", x=0.5),
        paper_bgcolor='white',  
        plot_bgcolor='white', 
        xaxis=dict(title=dict(text=label_x, font=dict(color='black')), tickfont=dict(color='black'),showline=True),
        yaxis=dict(title=dict(text=label_y, font=dict(color='black')), tickfont=dict(color='black'),showline=True),
        hovermode='x unified', # Info-bulle commune sur l’axe X
        dragmode='zoom', # Activation du zoom interactif par sélection
        margin=dict(t=100, b=10, l=10, r=10) # marges reduites
    )
    fig.update_traces(text=None, hoverinfo='name+x+y')

    return fig



#Fonction pour créer le tableau des métriques
def creation_tableau (df_kpi_selection):
    """
    Crée un tableau interactif Plotly affichant les indicateurs de performance (KPI) 
    pour chaque modèle sélectionné.

    Si aucun KPI n’est disponible, la fonction affiche un tableau contenant un message d’information.
    Sinon, les données sont pivotées pour présenter chaque indicateur en colonne.

    Paramètre :
        df_kpi_selection (pd.DataFrame) : Données KPI filtrées sur la sélection des modeles de prédictions

    Résultat retourné :
        - go.Figure : Tableau Plotly affichant les métriques ou un message si aucun KPI n'est sélectionné.
    """

    # style commun
    style_header = dict(
        fill_color='white',  # Fond blanc
        font=dict(color='rgb(24,112,184)', size=12), 
        align='left',
        height=30,
        line=dict(color='black', width=2)
    )
    style_cells = dict(
        fill_color='white',  
        font=dict(color='black', size=12),
        align='left',
        height=25,
        line=dict(color='black', width=2)
    )

    # Affiche un message si aucune donnée associée aux KPI
    if df_kpi_selection.empty:
        tab = go.Figure(data=[go.Table(
            header=dict(
                values=["\u00A0"],  # Espace insécable pour éviter "None"
                **style_header
            ),
            cells=dict(
                values=[["Aucune donnée sélectionnée pour les métriques."]],
                **style_cells
            )
        )])
        
        return tab
    
    # Pivot pour avoir les indicateurs en colonnes 
    df_kpi_pivot = df_kpi_selection.pivot(index='nom donnee', columns='indicateur', values='valeur').reset_index()


    # Création du tableau 
    tab = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>{}</b>'.format(col.upper()) for col in df_kpi_pivot.columns],  # Mise en gras et majuscule des titres
            **style_header
        ),
        cells=dict(
            values=[df_kpi_pivot[col] for col in df_kpi_pivot.columns],  # Remplissage des cellules
            **style_cells
        )
    )])
    tab.update_layout(
        autosize=False,  # Désactive l'ajustement automatique
        height=350,
        margin=dict(t=0, l=0, r=0, b=0)  # Suppression totale des marges pour éviter tout centrage vertical
    )
 
    return tab 


    

