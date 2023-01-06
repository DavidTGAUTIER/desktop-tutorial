import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy
import plotly.figure_factory as ff
import plotly.io as pio
from google.oauth2 import service_account
from google.cloud import storage
import os

st.set_page_config(
    page_title = "Get Around Project",
    page_icon = " ",
    layout = "wide"
)

st.header("Analyse des prix des locations")

st.markdown("""Dashboard pour le projet GetAround : pricing.

Ceci est la seconde partie concernant une étude le prix des locations. Nous avons un dataset qui contient toutes les informations sur les véhicules de location : marque, puissance du moteur, couleur de la peinture, ect...

En fonction de ces paramètres, Get Around determine un prix de location du véhicule. Nous allons donc analyser ces prix pour ensuite faire des prédictions

Voici en détail les colonnes du dataframe `pricing`:
* **`model_key`** : Marque du véhicule emprunté
* **`mileage`** : Nombre de kms
* **`engine_power`** : Puissance du moteur
* **`fuel`** : type de carburant
* **`paint_color`** : couleur du véhicule
* **`car_type`** : type de véhicule (sport, Van, coupé,...)
* **`private_parking_available`** : si parking privé disponible (`Oui` ou `Non`)
* **`has_gps`** : possède un GPS (`Oui` ou `Non`)
* **`has_air_conditioning`** : possède un climatiseur (`Oui` ou `Non`)
* **`automatic_car`** : le véhicule est il automatique (`Oui` ou `Non`)
* **`has_getaround_connect`** : possède l'option **GetAround Connect** (permet un check-in sans rencontre entre proprio / locataire) (`Oui` ou `Non`)
* **`has_speed_regulator`** : possède un régulateur de vitesse (`Oui` ou `Non`)
* **`winter_tires`** : possède des pneus d'hiver (pour la neige) (`Oui` ou `Non`)
* **`rental_price_per_day`** : prix de la location pour la journée (Target)

A la fin de cette partie, il y a une partie optionnel sur l'etude d'un seuil optimal optimal pour eviter les retards et en même temps, eviter de perdre de l'argent en retardant les locations entre chauffeurs""")

st.sidebar.write("Dashboard made by [@DavidT](https://github.com/DavidTGAUTIER)")
# st.sidebar.success("Navigation")

st.markdown("""
    ------------------------
""")

st.subheader("Chargement des données")

aws=False
local=True

data_load_state = st.text('Chargement des données...')

if aws:

    credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
    client = storage.Client(credentials=credentials)
    # permet de memorizer les executions de fonctions (cad les mettre en cache pour eviter d'avoir besoin de les relancer : on choisi une valeur de 900s, cad que si pendant 15min il y a un changement, charge auto la fonction)
    @st.experimental_memo(ttl=600)
    def import_data(bucket_name, file_path):
        bucket = client.bucket(bucket_name)
        content = bucket.blob(file_path).download_as_string().decode('utf-8')
        return content

    bucket_name = "get-around-bucket"
    file_path = "pricing.csv"

    data = import_data(bucket_name, file_path)
    data = pd.read_csv(io.StringIO(data))

if local:
    # Importation des données depuis environnement local
    @st.cache(allow_output_mutation=True)
    def import_data(path):
        data = pd.read_csv(path)
        return data

    path_normal = './src/pricing_cleaned.csv'
    path_outliers = './src/pricing_without_outliers.csv'

    data = import_data(path_normal)
    df_without_outliers = import_data(path_outliers)

data_load_state.text("Données disponibles")

# Show raw data
if st.checkbox('Montrer les données brutes'):
    st.subheader('Données brutes')
    st.write(data)

st.markdown(f"Ce jeu de donnée contient {data.shape[0]} lignes et {data.shape[1]} colonnes")

st.markdown("""
    ------------------------
""")



st.subheader("Repartition generale des données en fonction des paramètres du véhicule")

data = data[(data['mileage'] > 0) & (data['mileage']  < 500000)]

liste_engine = ['0-50','50-100','100-150','150-200','200-250','250-300','300-350','350-400','> 400']
liste_mileage = ['0-50000','50000-100000','100000-150000','150000-200000','200000-250000','250000-300000','300000-350000','350000-400000','> 400000']
def create_categ_engine(x, liste):
    for el in liste:
        if 0<x<50:
            return liste[0]
        elif 50<=x<100:
            return liste[1]
        elif 100<=x<150:
            return liste[2]
        elif 150<=x<200:
            return liste[3]
        elif 200<=x<250:
            return liste[4]
        elif 250<=x<300:
            return liste[5]
        elif 300<=x<350:
            return liste[6]
        elif 350<=x<400:
            return liste[7]
        else:
            return liste[8]

def create_categ_mileage(x, liste):
    for el in liste:
        if 0<x<50000:
            return liste[0]
        elif 50000<=x<100000:
            return liste[1]
        elif 100000<=x<150000:
            return liste[2]
        elif 150000<=x<200000:
            return liste[3]
        elif 200000<=x<250000:
            return liste[4]
        elif 250000<=x<300000:
            return liste[5]
        elif 300000<=x<350000:
            return liste[6]
        elif 350000<=x<400000:
            return liste[7]
        else:
            return liste[8]
        
data['categ_engine'] = data['engine_power'].map(lambda x:create_categ_engine(x,liste_engine))
data['categ_mileage'] = data['mileage'].map(lambda x:create_categ_mileage(x,liste_mileage))

col1, col2 = st.columns(2)

colors = ['cyan','royalblue', 'darkblue', 'lightcyan', 'mediumturquoise', 'lightblue', 'blue']

with col1:
    st.markdown("Choisir un type de statut ")
    statut = st.selectbox("Selectionnez le type de statut", ['fuel', 'type', 'model'])
    if statut == 'fuel':
        path, labels = ['fuel', 'paint_color', 'car_type', 'model_key'], {'fuel':'checkin_type'}
        values='rental_price_per_day'
        md = "On observe une très grande majorité de véhicule fonctionnant avec du Diesel comme type de carburant. On remarque aussi une majorité de véhicules ont des couleurs sombres (noir ou gris)."
    elif statut == 'type':
        path, labels = ['car_type', 'model_key'], {'type':'checkin_type'}
        values='engine_power'
        md = "En fonction du véhicule, en prenant en compte la puissance du moteur, on voit que ceux de marque Française sont les plus puissants (Citroen, Renault) suivi par des marques allemandes (Audi). Il y a une bonne repartition en fonction du type de voitures pour `estate`, `SUV` et `sedan`."
    else:
        path, labels = ['model_key', 'car_type'], {'model':'checkin_type'}
        values='rental_price_per_day'
        md = "Comme nous l'avions vu précédemment, ce sont les véhicules Français qui sont le plus loués (`Renault`, `Citroen`) avec comme type `sedan` et `estate`."
    fig1 = px.sunburst(data, path=path, values=values, 
                     color_discrete_sequence=colors, width=650,height=650)
    fig1.update_yaxes(title=f" Prix des locations en fonction de {statut} ")
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown(md)

with col2:
    st.markdown("Choisir un type de statut ")
    s= st.selectbox("Selectionnez le type de statut", ['fuel_type', 'engine_type', 'model_type'])
    if s == 'fuel_type':
        df = data
        col = 'fuel'
        values='rental_price_per_day'
        md = [(df['fuel'].value_counts().index[i],key) for i,key in enumerate(df['fuel'].value_counts())]
    elif s == 'engine_type':
        df = data
        col = 'categ_engine'
        values='rental_price_per_day'
        md = [(df['categ_engine'].value_counts()[:6].index[i],key) for i,key in enumerate(df['categ_engine'].value_counts()[:6])]
    else:
        df = data[(data['model_key']=='Citroën')|(data['model_key']=='Renault')|(data['model_key']=='BMW')|(data['model_key']=='Peugeot')|(data['model_key']=='Audi')|(data['model_key']=='Nissan')]
        col = 'model_key'
        values='rental_price_per_day'
        md = [(df['model_key'].value_counts()[:6].index[i],key) for i,key in enumerate(df['model_key'].value_counts()[:6])]
    fig2 = px.histogram(x=df[values], color=df[col], color_discrete_sequence=colors, width=650,height=650)
    fig2.update_yaxes(title=f" Distribution du prix des locations en fonction de {s} ")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(md)

st.markdown(" Nous allons observer la repartition des données sur les variables numériques (et voir les outliers) ")

st.subheader("Variables numériques : `mileage`, `engine_power` et `rental_price_per_day`")

st.markdown(" Sur les trois variables numériques de notre dataset, il y a la target qui est le prix des locations. Les deux autres variables sont la puissance du moteur et le nombre de miles effectués par le véhicule ")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(""" Distribution de la variable `mileage` """)
    fig1 = ff.create_distplot([data['mileage'].sample(500).dropna().sort_values().values], group_labels=['mileage'])
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown(""" Distribution de la variable `rental_price_per_day` """)
    fig2 = ff.create_distplot([data['rental_price_per_day'].dropna().sort_values().values], group_labels=['rental_price_per_day'])
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    st.markdown(""" Distribution de la variable `engine_power` """)
    fig3 = ff.create_distplot([data['engine_power'].dropna().sort_values().values], group_labels=['engine_power'])
    st.plotly_chart(fig3, use_container_width=True)

st.markdown(""" Concernant les variables numériques, on remarque une certaine homogénité des données. 

Pour la variable `mileage`, la plupart des données sont situées entre 100k et 300k miles, ce qui correspond à des vehicules qui ont déja bien été utilisé, ce qui s'explique par un prix de location plus avantageux.

Pour la variable  `rental_price_per_day`, on voit une belle courbe gaussienne centrée en 120$ avec un très léger etalement vers la droite, ce qui signifit que les locataires des véhicules sont prés à payer un peu plus cher que les prix moyens.

Enfin, on décèle ,pour la variable `engine_power`, deux pics à 120 et 135 qui corresponde à une valeur standard de puissance de véhicule : les personnes recherchent des voitures sobres, utiles.

On va regarder les boxplots pour se rendre compte de possibles outliers déja repérer pendant l'etude sur notebook des prix des locations""")

plot_rows=1
plot_cols=3
fig = make_subplots(rows=plot_rows, cols=plot_cols, subplot_titles=("Mileage","Rental price per day", "Engine power"))

# plotly traces
fig.add_trace(go.Box(y=data['mileage'], marker_color='royalblue'), row=1, col=1)
fig.add_trace(go.Box(y=data['rental_price_per_day'], marker_color='cyan'), row=1, col=2)
fig.add_trace(go.Box(y=data['engine_power'], marker_color='darkblue'), row=1, col=3)
fig.update_layout(
    showlegend=False,
    title_text="Boxplots des variables numériques")
st.plotly_chart(fig, use_container_width=True)

st.markdown("""Il existe de nombreux outliers pour ces trois variables.

 Pour `mileage`, il existait un outlier situé au dessus des 1 millions de km, ce qui est extrement rare ou carrément improbable, que nous avons décidé de supprimer avant de lancer l'application.
De plus, on remarque également qu'il y a beaucoup de véhicules avec plus de 300.000km qui sont en dehors du boxplot, ce qui signifit que les locataires de véhicules louent rarement des voitures trop vieille ou avec trop de kms.

Avec la variable `rental_price_per_day`, la distribution est a peu près identique à celle de `engine_power` : une moyenne située aux alentours de 120, une médiane équivalente à la moyenne (donc une distribution gaussienne) et des outliers en dessous des 60 et au dessus des 180.

Analysons maintenant un peu plus en détails ces trois variables.""")

plot_rows=3
plot_cols=3
fig = make_subplots(rows=plot_rows, cols=plot_cols, subplot_titles=("Mileage","Rental price per day", "Engine power"))

# plotly traces
fig.add_trace(go.Histogram(x=data['mileage'], marker_color='royalblue'), row=1, col=1)
fig.add_trace(go.Scatter(x=data['mileage'].sort_values(), y=data['rental_price_per_day'], mode='markers', marker_color='royalblue'), row=2, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['mileage'].value_counts(), marker_color='royalblue'), row=3, col=1)
fig.add_trace(go.Histogram(x=data['rental_price_per_day'], marker_color='cyan'), row=1, col=2)
fig.add_trace(go.Scatter(x=data['rental_price_per_day'].sort_values(), y=data['rental_price_per_day'], mode='markers', marker_color='cyan'), row=2, col=2)
fig.add_trace(go.Scatter(x=data.index, y=data['rental_price_per_day'].value_counts(), marker_color='cyan'), row=3, col=2)
fig.add_trace(go.Histogram(x=data['engine_power'], nbinsx=50, marker_color='darkblue'), row=1, col=3)
fig.add_trace(go.Scatter(x=data['engine_power'].sort_values(), y=data['rental_price_per_day'], mode='markers', marker_color='darkblue'), row=2, col=3)
fig.add_trace(go.Scatter(x=data.index, y=data['engine_power'].value_counts(), marker_color='darkblue'), row=3, col=3)
fig.update_layout(
    showlegend=False,
    title_text="Distributions des variables numériques")
st.plotly_chart(fig, use_container_width=True)

st.markdown(""" De manière plus détaillée, on peut observer des distributions similaires, autant en terme de dispersion que de regroupement de données. 

Nous allons maintenant supprimer les outliers et ne garder que des ranges représentant un peu mieux la repartition des données.

Regardons à présent les variables catégorielles""")

st.subheader("Variables catégorielles : `model_key`, `fuel`, `paint_color` et `car_type`")

st.markdown(" Il y a donc quatre variables catégorielles : la marque du véhicule, son type de carburant, sa couleur ainsi que le type de véhicule. Nous allons analyser variable par variable puis en comparant ces variables avec la target")

def make_categ_subplots(data, col, subplot_titles, title_text, vertical_spacing):
    plot_rows=3
    plot_cols=2
    fig = make_subplots(rows=plot_rows, cols=plot_cols, subplot_titles=subplot_titles, vertical_spacing=vertical_spacing)

    # plotly traces
    fig.add_trace(go.Bar(x=data[col].unique(), y=data[col].value_counts(), marker_color='royalblue'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data[col].sort_values(), y=data['rental_price_per_day'], mode='markers', marker_color='royalblue'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data[col].value_counts(), marker_color='royalblue'), row=3, col=1)

    fig.add_trace(go.Bar(x=data.groupby('model_key')['rental_price_per_day'].mean().sort_values(ascending=False).index, y=data.groupby(col)['rental_price_per_day'].mean().sort_values(ascending=False), marker_color='cyan'), row=1, col=2)
    fig.add_trace(go.Scatter(x=data.groupby(col)['rental_price_per_day'].sum(), y=data['rental_price_per_day'], mode='markers', marker_color='cyan'), row=2, col=2)
    fig.add_trace(go.Scatter(x=data.index, y=data.groupby(col)['rental_price_per_day'].mean().sort_values(ascending=False), marker_color='cyan'), row=3, col=2)

    fig.update_layout(
        showlegend=False,
        title_text=title_text)
    return fig

def make_categ_bar_subplots(data, col):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data[col].unique(), y=data[col].value_counts(), marker_color='royalblue', name='le plus loué'))
    fig.add_trace(go.Scatter(x=data[col].unique(), y=data[col].value_counts(), marker_color='royalblue', showlegend=False, fill='tonexty'))
    fig.add_trace(go.Bar(x=data.groupby(col)['rental_price_per_day'].mean().sort_values(ascending=False).index, y=data.groupby(col)['rental_price_per_day'].mean().sort_values(ascending=False), marker_color='cyan', name='le plus rentable'))
    fig.add_trace(go.Scatter(x=data[col].unique(), y=data.groupby(col)['rental_price_per_day'].mean()[data[col].unique()].values, marker_color='cyan', showlegend=False,fill='tozeroy'))
    return fig

st.markdown("Choisir un type de statut ")
s= st.selectbox("Selectionnez le type de statut", ['marque du vehicule', 'type de carburant', 'couleur de la peinture', 'type du vehicule'])
if s == 'marque du vehicule':
    col = 'model_key'
    titles = ("Marque de voiture les plus louées", "Marque de voiture les plus rentables")
    title_text = "Marque de voitures"
    vertical_spacing = 0.30
    md = "Les 5 marques les plus représentées sont : **Citroen**, **Renault**, **BMW**, **Peugeot** et **Audi**. On observe que les marques des véhicules les plus louées sont les moins cher en général"
elif s == 'type de carburant':
    col = 'fuel'
    titles = ("Type de carburant des véhicules les plus loués", "Type de carburant des véhicules les plus rentables")
    title_text ="Type de carburant"
    vertical_spacing = 0.15
    md = "On retrouve une sur-représentation du type de carburant `Diesel` dans les véhicules les plus loués. Par contre, les véhicules les plus rentables sont bien de type `Electrique` mais ils sont sous représentés (seulement 3 véhicules electrique de loués dans le dataset)"
elif s == 'couleur de la peinture':
    col = 'paint_color'
    titles = ("Couleur de la peinture des véhicules les plus loués", "Couleur de la peinture des véhicules les plus rentables")
    title_text = "Type de peinture"
    vertical_spacing = 0.15
    md = "Les couleurs des véhicules les plus loués sont assez sombres (`black`, `grey`) alors que les couleurs les plus rentables sont plus vive/claire comme `orange`, `white` ou `beige` (qui peut être lié au voiture de sport qui sont souvent de couleur vive et qui sont donc plus chers à la location)"
else:
    col = 'car_type'
    titles = ("Type de modèle des véhicules les plus loués", "Type de modèle des véhicules les plus rentables")
    title_text = "Type de véhicules"
    vertical_spacing = 0.15
    md = "En ce qui concerne le type de modèle des véhicules, les types les plus loués (`estate`, `sedan`) correspondent à des voitures plutôt classiques alors que les plus rentables (`coupe`, `convertible`) sont plus typé **sport**. Le type `SUV` est autant loué que rentable."
fig = make_categ_subplots(df_without_outliers, col=col, subplot_titles=titles, title_text=title_text, vertical_spacing=vertical_spacing)
st.plotly_chart(fig, use_container_width=True)
st.markdown(md)

st.markdown("Regardons le type de distributions un peu plus en détail")

col1, col2 = st.columns(2)

with col1:
    s1 = st.selectbox("Selectionnez le type de statut en fonction des locations", ['marque du vehicule', 'type de carburant', 'couleur de la peinture', 'type du vehicule'])
    if s1 == 'marque du vehicule':
        col = 'model_key'
        title = "Marque de voiture les plus louées"
    elif s1 == 'type de carburant':
        col = 'fuel'
        title = "Type de carburant des véhicules les plus loués"
    elif s1 == 'couleur de la peinture':
        col = 'paint_color'
        title = "Couleur de la peinture des véhicules les plus loués"
    else:
        col = 'car_type'
        title = "Type de modèle des véhicules les plus loués"
    fig1 = px.bar(df_without_outliers[col].value_counts(), title=title)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    s2 = st.selectbox("Selectionnez le type de statut en fonction de la rentabilité", ['marque vehicule', 'type carburant', 'couleur peinture', 'type vehicule'])
    if s2 == 'marque vehicule':
        col = 'model_key'
        title = "Marque de voiture les plus rentables"
    elif s2 == 'type carburant':
        col = 'fuel'
        title = "Type de carburant des véhicules les plus rentables"
    elif s2 == 'couleur peinture':
        col = 'paint_color'
        title = "Couleur de la peinture des véhicules les plus rentables"
    else:
        col = 'car_type'
        title = "Type de modèle des véhicules les plus rentables"
    fig2 = px.bar(df_without_outliers.groupby(col)['rental_price_per_day'].mean().sort_values(ascending=False), title=title)
    st.plotly_chart(fig2, use_container_width=True)

stat = st.selectbox("Représentation générale des variables categorielles", ['marque du vehicule', 'type de carburant', 'couleur de la peinture', 'type du vehicule'])
if stat == 'marque du vehicule':
    col = 'model_key'
elif stat == 'type de carburant':
    col = 'fuel'
elif stat == 'couleur de la peinture':
    col = 'paint_color'
else:
    col = 'car_type'
fig = make_categ_bar_subplots(df_without_outliers, col)
st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Conclusion de l'etude sur les variables categorielles")
st.markdown(""" On peut en conclure qu'il y a bien une différence entre les variables catégorielles. Souvent, les variables les plus loués sont les moins chers. De plus, on remarque que les véhicules les plus loués sont sobres et pris pour leur utilité et leur faible cout de revient. Alors que les véhicules les plus chers sont plus pour le coté plaisir, de couleurs vives, de type électrique et sportive. """)

st.subheader("Variables booléennes : `private_parking_available`, `has_gps`, `has_air_conditioning`, `automatic_car`, `has_getaround_connect`, `has_speed_regulator` et `winter_tires`")

st.markdown("""Le dernier type de variables correspond aux booleens : cad qu'il n'y a que deux réponses possibles : **Oui** ou **Non**

Ce type de variable concerne les options des véhicules qui vont augmenter le prix du véhicule : plus il y a de `True`, plus le prix augmente.

On va regarder les distributions de ces variables en fonction du prix final de location car elles ne nous donnent que trés peu de diversité dans leur représentation""")

s=[]
for col in df_without_outliers.select_dtypes(bool):
    print(col)
    s.append(col)

num_cols = df_without_outliers.select_dtypes(bool).shape[1]
num_rows = 1
i = 0
specs = np.repeat({'type':'pie'}, num_cols).reshape(num_rows, num_cols).tolist()
colors=['mediumturquoise', 'royalblue']
fig = make_subplots(rows=num_rows, cols=num_cols, specs=specs)
for col in df_without_outliers.select_dtypes(bool):
    no_df = df_without_outliers[df_without_outliers[col]==False].shape[0]
    yes_df = df_without_outliers[df_without_outliers[col]==True].shape[0]
    fig.add_trace(go.Pie(labels=['No', 'Yes'], values=[no_df, yes_df], name=col), num_rows, i+1)
    i+=1
fig.update_layout(
    title_text=" Proportion des Variables boolean ",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text=s[0], x=-0.03, y=-0.17, font_size=12, showarrow=False, textangle=-45),
                 dict(text=s[1], x=0.18, y=0.06, font_size=12, showarrow=False, textangle=-45),
                 dict(text=s[2], x=0.29, y=-0.11, font_size=12, showarrow=False, textangle=-45),
                 dict(text=s[3], x=0.48, y=-0.02, font_size=12, showarrow=False, textangle=-45),
                 dict(text=s[4], x=0.61, y=-0.14, font_size=12, showarrow=False, textangle=-45),
                 dict(text=s[5], x=0.80, y=-0.11, font_size=12, showarrow=False, textangle=-45),
                 dict(text=s[6], x=0.95, y=-0, font_size=12, showarrow=False, textangle=-45),
                ])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=12,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    
st.plotly_chart(fig, use_container_width=True)

fig = px.sunburst(df_without_outliers, path=['model_key']+s, values='rental_price_per_day',width=800,height=800, 
labels={'rental_price_per_day':'Rental revenue per day'}, color_discrete_sequence=['cyan','royalblue', 'darkblue'])
st.plotly_chart(fig, use_container_width=True)

st.markdown("""En ce qui concerne la repartition des `True` / `False`, on remarque qu'il y a plus de `False` que de `True` en général (plus le type de véhicule coute cher à la location, plus il y a de `True` (exemple: **Audi** ou **BMW**)""")
