import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy
import plotly.figure_factory as ff
import plotly.io as pio
import copy
import boto3
import os

st.set_page_config(
    page_title = "Get Around Project",
    page_icon = " ",
    layout = "wide"
    )

st.title("Projet Get Around")
st.header("Analyse des retards")
st.markdown("""Dashboard pour le projet GetAround : delays.

Il contient une analyse des durées de deplacements des chauffeurs utilisant l'application GetAround. 
Comme certains chauffeurs rendent les véhicules en retard, le but de ce projet est de mettre en place un **délai minimum entre deux locations** : \n
un véhicule ne s'affichera pas dans les résultats de recherche si les heures d'enregistrement ou de départ demandées sont trop proches d'une location déjà réservée **sans que cela pénalise financièrement les propriétaires de ces véhicules**.

Nous avons choisi de diviser en deux parties cette analyse:
la première partie concerne les retards des chauffeurs alors que la seconde partie couvre les types de véhicules et le prix d'une location.""")

st.markdown("""voici les colonnes du dataframe `delay_analysis`:
* **`rental_id`** : identifiant de la location
* **`car_id`** : identifiant de la voiture
* **`checkin_type`** : type de location :
    * **mobile** : propriétaire / locataire se rencontrent et signent ensemble sur le smartphone du propriétaire
    * **connect** : pas de rencontre entre propriétaire / locataire (le locataire ouvre le vehicule avec son smartphone)
* **`state`** : etat de location :
    * **canceled** : location annulée (on ne connait pas la raison, peut être délais trop important, propriétaire qui a besoin de son véhicule,ect..)
    * **ended** : location terminée (le locataire à rendu le véhicule au propriétaire)
* **`delay_at_checkout_in_minutes`** : délais de retard en minutes
* **`previous_ended_rental_id`** : précédent identifiant de location 
* **`time_delta_with_previous_rental_in_minutes`** : temps de la précédente location en minutes""")


st.sidebar.write("Dashboard made by [@DavidT](https://github.com/DavidTGAUTIER)")
#st.sidebar.success("Navigation")

st.markdown("""
    ------------------------
""")

st.subheader("Chargement des données")


aws=False
local=True

if aws:
    @st.cache
    # Importation des données depuis AWS s3
    def import_data():

        client = boto3.client(
            "s3",
            # il faut d'abord crée les variables d'environnement "s3_key" et "s3_secret" : s3_key = XXXXXX, s3_secret = XXXXXX, après on peut les appeler avec os.getenv(key)
            aws_access_key_id = os.getenv("AWS_S3_KEY"),
            aws_secret_access_key=os.getenv("AWS_S3_SECRET")
        )

        response = client.get_object(Bucket = "get-around-bucket", Key = "delays_cleaned.csv")
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

        if status != 200:
            return f"Erreur de connexion avec AWS s3 - Status code :{status}"

        else:
            return pd.read_csv(response.get("Body"))

if local:
    @st.cache
    # Importation des données depuis environnement local
    def import_data():
        data = pd.read_csv('./src/delays_cleaned.csv')
        return data

data_load_state = st.text('Chargement des données...')
data = import_data()
data_load_state.text("Données disponibles")

# Show raw data
if st.checkbox('Montrer les données brutes'):
    st.subheader('Données brutes')
    st.write(data)

st.markdown(f"Ce jeu de donnée contient {data.shape[0]} lignes et {data.shape[1]} colonnes")

st.markdown("""
    ------------------------
""")

data_ended = data.loc[data['state']=='ended',:]
data_cancel = data.loc[data['state']=='canceled',:]
data_mobile = data.loc[data['checkin_type']=='mobile',:]
data_connect = data.loc[data['checkin_type']=='connect',:]

st.subheader("Repartition des données en fonction du statut d'une course et du type de check_in")

col1, col2 = st.columns(2)

with col1:
    st.markdown("Choisir un type de statut ")
    statut = st.selectbox("Selectionnez le type de statut", ['canceled & ended', 'ended', 'canceled'])
    if statut == 'canceled & ended':
        data = data
        path, labels = ['state', 'checkin_type'], {'state':'checkin_type'}
        colors = ['cyan','royalblue']
        checkin = 'annulées et terminées'
    elif statut == 'ended':
        data = data_ended
        path, labels = ['state', 'checkin_type'], {'state':'checkin_type'}
        colors = ['cyan','royalblue']
        checkin = 'terminées'
    else:
        data = data_cancel
        path, labels = ['state', 'checkin_type'], {'state':'checkin_type'}
        colors = ['royalblue', 'cyan']
        checkin = 'annulées'
    fig = px.sunburst(data, path=path, values='time_delta_with_previous_rental_in_minutes', 
                     color_discrete_sequence=colors, width=600,height=600)
    fig.update_yaxes(title=f" Courses {checkin} en fonction du type de statut ")
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Nombre de samples : ", len(data))

with col2:
    st.markdown("Choisir un type de checkin")
    statut2 = st.selectbox("Selectionnez le type de checkin", ['mobile & connect', 'mobile', 'connect'])
    if statut2 == 'mobile & connect':
        data = data
        path, labels2 = ['checkin_type', 'state'], {'checkin_type':'state'}
        colors = ['cyan','royalblue']
        checkin = 'mobile et connect'
    elif statut2 == 'mobile':
        data = data_mobile
        path, labels = ['checkin_type', 'state'], {'checkin_type':'state'}
        colors = ['cyan','royalblue']
        checkin = 'mobile'
    else:
        data = data_connect
        path, labels = ['checkin_type', 'state'], {'checkin_type':'state'}
        colors = ['cyan', 'royalblue']
        checkin = 'connect'
    fig2 = px.sunburst(data, path=path, values='time_delta_with_previous_rental_in_minutes', 
                     labels=labels, color_discrete_sequence=colors, width=600,height=600)
    fig2.update_yaxes(title=f" Courses en fonction du type de check-in : {checkin}")
    st.plotly_chart(fig2, use_container_width=True)
    st.metric("Nombre de samples : ", len(data))

st.markdown(""" Le dataset contient plus de courses terminées que de courses annulées. Les courses annulées ont le même ration de check-in `mobile` ou `connect` alors que celles qui se sont bien terminées ont plus de check-in de type 'mobile'.

Regardons à présent plus en détail les retards sur ces courses""")

st.markdown("""
    ------------------------
""")

st.subheader("Analyse des retards par catégories de tranches horaire en fonction du type de check-in")

fig = go.Figure()
colors = ['cyan','royalblue', 'darkblue']
fig.add_trace(go.Bar(x=data_mobile['late_delay'].value_counts().index, y=data_mobile['late_delay'].value_counts(), name='mobile', marker_color='royalblue'))
fig.add_trace(go.Bar(x=data_connect['late_delay'].value_counts().index, y=data_connect['late_delay'].value_counts(), name='connect', marker_color='cyan'))
fig.update_layout(
    title_text="Repartition des retards en fonction du type de check-in (format Bar)",
    barmode="stack",
    uniformtext=dict(mode="hide", minsize=10),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(""" Nous pouvons observer que la majorité des retards sont soit plus long que 2heures soit durent moins de 15 minutes. Les retards problématiques sont ceux d'une durée trop importante car ils peuvent occasionner une annulation de la prochaine course.""")

st.subheader("Repartition des retards par catégories de tranches horaire en fonction du statut de retard ")

col1, col2 = st.columns(2)
df = data[['checkin_type', 'state', 'delay_at_checkout_in_minutes', 'late_delay', 'time_delta_with_previous_rental_in_minutes']].dropna()

with col1:
    st.markdown(""" Retard avec vue simplifiée """)
    fig1 = px.sunburst(df, path=['checkin_type', 'state', 'late_delay'], values='time_delta_with_previous_rental_in_minutes',width=700,height=700, 
    labels={'delais':'time_delta'}, color_discrete_sequence=['cyan','royalblue'])
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown(""" Retard avec vue détaillée""")
    fig2 = px.sunburst(df, path=['checkin_type', 'state', 'late_delay', 'delay_at_checkout_in_minutes'], values='time_delta_with_previous_rental_in_minutes',width=700,height=700, 
    labels={'delais':'time_delta'}, color_discrete_sequence=['cyan','royalblue'])
    st.plotly_chart(fig2, use_container_width=True)

st.markdown(""" On observe également des retards plus long pour le type de check-in mobile que celui connect : cela peut être du à deux raisons : 
la première est qu'il y a moins de clients qui prennent un type de check-in mobile, la seconde raison est que ces clients ne vont pas rencontrer en personne le propriétaire ou le nouveau locataire donc ils ont moins de pression à ramener le véhicule à l'heure""")

st.markdown("""
    ------------------------
""")

st.subheader("Analyse de la répertition des retards (avec et sans outliers)")

data_early = data[(data['late_delay']=='Early')]
data_late = data[(data['late_delay']!='Early')]

st.markdown("#### Avec Outliers")

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'box'}, {'type':'box'}]], subplot_titles=("Outliers(log)","Outliers(linear)"))
fig.add_trace(go.Box(y=data_late["delay_at_checkout_in_minutes"], name='All delays', quartilemethod='linear', boxmean='sd', boxpoints='suspectedoutliers', marker_color='royalblue'),1,1)
fig.update_yaxes(type="log", row=1, col=1)
fig.update_traces(quartilemethod="linear", jitter=0)
fig.add_trace(go.Box(y=data_late["delay_at_checkout_in_minutes"], name='All delays', quartilemethod='linear', boxmean='sd', boxpoints='suspectedoutliers', marker_color='cyan'),1,2)
fig.update_yaxes(type="linear", row=1, col=2)
st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Sans Outliers")

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'box'}, {'type':'box'}]], subplot_titles=("Without Outliers > 20k(log)","Without Outliers > 20k(linear)"))
fig.add_trace(go.Box(y=data[(data['delay_at_checkout_in_minutes']<20000) & (data['late_delay']!='Early')]["delay_at_checkout_in_minutes"], name='delays < 20000', quartilemethod='linear', boxmean='sd', boxpoints='suspectedoutliers', marker_color='royalblue'),1,1)
fig.update_yaxes(type="log", row=1, col=1)
fig.update_traces(quartilemethod="linear", jitter=0)
fig.add_trace(go.Box(y=data[(data['delay_at_checkout_in_minutes']<20000) & (data['late_delay']!='Early')]["delay_at_checkout_in_minutes"], name='delays < 20000', quartilemethod='linear', boxmean='sd', boxpoints='suspectedoutliers', marker_color='cyan'),1,2)
fig.update_yaxes(type="linear", row=1, col=2)
st.plotly_chart(fig, use_container_width=True)

st.markdown(""" On remarque des delais hors-norme qui s'etalent sur plus de 40 jours. Il reste quand même beaucoup d'outliers entre 4 et 13jours de retard ce qui vraiment trés long. Peut être est-ce du à un accident ...""")

st.markdown("""Nous allons maintenant analyser les différentes distributions en fonction du type de check-in sans ces outliers""")

st.markdown("""
    ------------------------
""")

st.subheader("Distribution des retards en fonction du type de check-in : `mobile` ou `connect` (sans outliers)")

remove_outliers = abs(data['delay_at_checkout_in_minutes'] - data['delay_at_checkout_in_minutes'].mean()) <= 2*data['delay_at_checkout_in_minutes'].std()
data_without_outliers = data.loc[remove_outliers, :]

fig = px.histogram(data_without_outliers, 'delay_at_checkout_in_minutes', nbins=100, color='checkin_type', barmode='overlay', marginal='box', color_discrete_sequence=['cyan','royalblue'])
st.plotly_chart(fig, use_container_width=True)

st.markdown("""Ces distributions sont quasi-identiques : `gaussiennes`, `centrées sur 0`. Par rapport aux tranches horaires, il y a donc autant de retards que d'avance dans le retour des locations. 
On remarque également qu'il y a plus de retards pour les check-in de type `mobile` entre 600 et 1800 minutes (ce qui correspond entre 10h et 30h) """)

st.subheader("Avance et retard des utilisateurs en fonction du type de controle : Mobile ou Connect")

col1, col2 = st.columns(2)

with col1:
    fig1= make_subplots(rows=1, cols=2, specs=[[{'type':'pie'}, {'type':'pie'}]], row_heights=[4])
    colors = ['cyan','royalblue']
    values_early = data[data['delay']=='early'].groupby('checkin_type')['delay_at_checkout_in_minutes'].sum().abs()
    values_lates = data[data['delay']=='late'].groupby('checkin_type')['delay_at_checkout_in_minutes'].sum()
    fig1.add_trace(go.Pie(labels=values_early.keys(), values=values_early, name="Chauffeur en avance", marker_colors=px.colors.qualitative.Prism), 1, 1)
    fig1.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20, marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    fig1.add_trace(go.Pie(labels=values_lates.keys(), values=values_lates, name="Chauffeur en retard", marker_colors=px.colors.sequential.Aggrnyl), 1, 2)
    fig1.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20, marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    fig1.update_traces(hole=.4, hoverinfo="label+percent+name")
    fig1.update_layout(annotations=[dict(text='En avance', x=0.14, y=0.5, font_size=20, showarrow=False), dict(text='En retard', x=0.855, y=0.5, font_size=20, showarrow=False)])
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.pie(data ,names='delay', facet_col='checkin_type', color_discrete_sequence=['cyan','royalblue'])
    st.plotly_chart(fig2, use_container_width=True)


st.markdown(""" On voit les mêmes phénomènes que sur l'histogramme précédent, à savoir que le type de check-in `mobile` cause plus de retards donc il serait peut être judicieux de faire payer plus cher ce type de prestations""")

st.markdown(""" Regardons les différences entre les courses terminées et celles qui ont été annulées """)

st.markdown("""
    ------------------------
""")

st.subheader("Distribution des retards en fonction du type de check-in : mobile ou connect (avec outliers)")

fig = make_subplots(rows=1, cols=2, subplot_titles=("Courses annulées","Courses terminées"))
fig.add_trace(go.Histogram(x=data_cancel[data_cancel['checkin_type']=='mobile']['time_delta_with_previous_rental_in_minutes'], marker_color='royalblue', name='mobile'),1,1)
fig.add_trace(go.Scatter(x=data_cancel[data_cancel['checkin_type']=='mobile']['time_delta_with_previous_rental_in_minutes'].sort_values().unique(), y=data_cancel[data_cancel['checkin_type']=='mobile']['time_delta_with_previous_rental_in_minutes'].value_counts().sort_index(), marker_color='royalblue', fill='tozeroy', showlegend=False),1,1)
fig.add_trace(go.Histogram(x=data_cancel[data_cancel['checkin_type']=='connect']['time_delta_with_previous_rental_in_minutes'],marker_color='cyan', name='connect'),1,1)
fig.add_trace(go.Scatter(x=data_cancel[data_cancel['checkin_type']=='connect']['time_delta_with_previous_rental_in_minutes'].sort_values().unique(), y=data_cancel[data_cancel['checkin_type']=='connect']['time_delta_with_previous_rental_in_minutes'].value_counts().sort_index(), marker_color='cyan', fill='tonexty', showlegend=False),1,1)
fig.add_trace(go.Histogram(x=data_ended[data_ended['checkin_type']=='mobile']['time_delta_with_previous_rental_in_minutes'], marker_color='royalblue', name='mobile', showlegend=False),1,2)
fig.add_trace(go.Scatter(x=data_ended[data_ended['checkin_type']=='mobile']['time_delta_with_previous_rental_in_minutes'].sort_values().unique(), y=data_ended[data_ended['checkin_type']=='mobile']['time_delta_with_previous_rental_in_minutes'].value_counts().sort_index(), marker_color='royalblue', fill='tonexty', showlegend=False),1,2)
fig.add_trace(go.Histogram(x=data_ended[data_ended['checkin_type']=='connect']['time_delta_with_previous_rental_in_minutes'],marker_color='cyan', name='connect', showlegend=False),1,2)
fig.add_trace(go.Scatter(x=data_ended[data_ended['checkin_type']=='connect']['time_delta_with_previous_rental_in_minutes'].sort_values().unique(), y=data_ended[data_ended['checkin_type']=='connect']['time_delta_with_previous_rental_in_minutes'].value_counts().sort_index(), marker_color='cyan', fill='tozeroy', showlegend=False),1,2)
st.plotly_chart(fig, use_container_width=True)

st.markdown("Choisir un type de state")
statut = st.selectbox("Selectionnez le type de statut", ['canceled', 'ended'])
if statut == 'canceled':
    data = data_cancel
else:
    data = data_ended
fig = px.histogram(data ,x='time_delta_with_previous_rental_in_minutes', nbins=100, color='checkin_type', color_discrete_sequence=['cyan','royalblue'])
fig.update_yaxes(title=f" Courses en fonction du type de statut : {statut}")
st.plotly_chart(fig, use_container_width=True)

st.markdown("""Les deux distributions sont similaires et nous ne pouvons pas conclure d'une différence significative.. en revanche, on se rend compte que la plupart des courses précédentes sont très courtes en terme de durée : cela chute de manière exponentielle et remonte légèrement après 400 minutes""")

st.markdown("""Regardons les sommes cumulées des retards pour observer une différence ou non""")

st.markdown("""#### Somme cumulée des retards""")

fig = px.histogram(data_ended, x='time_delta_with_previous_rental_in_minutes', histnorm='percent',cumulative=True, marginal='box', color = 'checkin_type', barmode='overlay', color_discrete_sequence=['cyan','royalblue'])
st.plotly_chart(fig, use_container_width=True)

st.markdown("""Il n'y a pas de différences entre le type de check-in `mobile` ou `connect` : les sommes cumulées sont bien réparties et montent de manière croissante (pas de pic exponentiel) en fonction de la durée des anciennes courses.""")

st.markdown("""Analysons cette fois-ci sans les outliers(cad sans les données qui sont plus de deux ecarts-types de la moyenne)""")

st.markdown("""
    ------------------------
""")

st.subheader("Distribution des retards en fonction du type de check-in : mobile ou connect (sans outliers)")

pairplot_df = data_without_outliers[['state','delay_at_checkout_in_minutes','time_delta_with_previous_rental_in_minutes']]
pairplot_df = pairplot_df.rename(columns={'delay_at_checkout_in_minutes':'delays_minutes', 'time_delta_with_previous_rental_in_minutes':'time_delta'})

col1, col2 = st.columns(2)

with col1:
    st.markdown(""" Retard des courses actuelles """)
    fig1 = ff.create_distplot([pairplot_df['delays_minutes'].dropna().sort_values().values], group_labels=['delays_minutes'])
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.markdown(""" Retards des courses passées""")
    fig2 = ff.create_distplot([pairplot_df['time_delta'].dropna().values], group_labels=['time_delta'])
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("""On voit une distribution normale comme celle que nous venons de voir, moyenne nulle et centrée sur 0. Nous avons choisi un range entre -2000 / 2000 minutes, ce qui correspond à +- 33 heures. 
En ce qui concerne les retards des courses passées, il y a seulement 27 valeurs pour ces courses, cela correspond à des types de seuils. Nous nous retrouvons avec un graphique en batons avec deux pics.

Observons plus en détail ces distributions""")

plot_rows=3
plot_cols=2
fig = make_subplots(rows=plot_rows, cols=plot_cols, subplot_titles=("Delays","TimeDelta"))

# plotly traces
fig.add_trace(go.Histogram(x=pairplot_df['delays_minutes'], marker_color='royalblue'), row=1, col=1)
fig.add_trace(go.Scatter(x=pairplot_df['delays_minutes'].sort_values(), y=pairplot_df['time_delta'], mode='markers', marker_color='royalblue'), row=2, col=1)
fig.add_trace(go.Histogram(x=pairplot_df['time_delta'].sort_values(), nbinsx=25, marker_color='cyan'), row=1, col=2)
fig.add_trace(go.Scatter(x=pairplot_df['time_delta'].sort_values(), y=pairplot_df['delays_minutes'], mode='markers', marker_color='cyan'), row=2, col=2)
fig.add_trace(go.Scatter(x=pairplot_df.index, y=pairplot_df['delays_minutes'].value_counts(), marker_color='royalblue'), row=3, col=1)
fig.add_trace(go.Scatter(x=pairplot_df.index, y=pairplot_df['time_delta'].value_counts(), marker_color='cyan'), row=3, col=2)
fig.update_layout(
    showlegend=False,
    title_text="Retards des anciennes et des nouvelles courses")
st.plotly_chart(fig, use_container_width=True)

st.markdown("""Il y a plus de valeurs centrées sur 0 pour les nouvelles que les anciennes courses. La distribution des anciennes courses est un peu plus 'chaotique', dans le sens ou il n'y a pas beaucoup de valeurs différentes, 
cette colonne contenant + 90% de valeurs manquantes n'aide pas à se faire une bonne représentation des retards passés. On observe bien deux pics : un aux alentours des 30-50 minutes de retard et le second vers 700min (11heures).

Observons la distribution des anciens retards plus en détails""") 

index = data['time_delta_with_previous_rental_in_minutes'].value_counts().index
values = data['time_delta_with_previous_rental_in_minutes'].value_counts().values

fig = px.bar(x=index, y=values, color_discrete_sequence=['royalblue'])
fig.update_yaxes(title='Nombre de chauffeurs en retard (count)')
fig.update_xaxes(title='Time delta')
st.plotly_chart(fig, use_container_width=True)

st.markdown("""Pour `time_delta_with_previous_rental_in_minutes`, on remarque un creux à 400 minutes, et deux ou trois pics au début et à la fin.

Si on mettait en place un seuil entre le départ et le nouvel enregistrement, combien de chauffeurs seraient concernés ?

Nous allons essayer de determiner un seuil pour répérer la frequence de chauffeurs qui rendent leur véhicule en retard après avoir pris un retard. Pour cela, nous prenons les valeurs de `time_delta_with_previous_rental_in_minutes` et on retranche la somme des valeurs pour créer des catégories de retards

On remarque egalement que plus les retard sont long, plus il y a de personnes qui rendent leur véhicule en retard.

On va créer plusieurs seuil en fonction du type de check-in : soit `mobile` soit `connect` pour se rendre compte celui qui a le plus d'influence sur les retards.
 Pour cela, on selectionne seulement les courses qui se sont bien terminées (qui n'ont pas été annulées). 
 Il faut savoir qu'il existe 26 valeurs de retards dans la catégorie `Time delta` (25 pour être exacte car le 26e est la valeur NaN).
  Pour chaque valeur (seuil) de retard, et pour chaque type de check-in(`mobile` ou `connect`), on compte le nombre de courses qui ont été en retard et on fait la somme cumulée pour avoir un graphique qui représente le nombre de courses affectées par ces retards
""")

st.markdown("""
    ------------------------
""")

st.subheader("Observation du seuil qui affecte les locations de véhicule en fonction du retard")

seuil_mobile, seuil_connect = [],[]
for el in data['time_delta_with_previous_rental_in_minutes'].unique():
    seuil_mobile.append(sum(data_mobile['time_delta_with_previous_rental_in_minutes'] < el)*100/len(data_ended))
    seuil_connect.append(sum(data_connect['time_delta_with_previous_rental_in_minutes'] < el)*100/len(data_ended))
seuil_total = [x+y for x, y in zip(seuil_mobile, seuil_connect)]

plot_rows=1
plot_cols=2
newnames = {'wide_variable_0':'mobile', 'wide_variable_1': 'connect', 'wide_variable_2': 'sum'}
fig = make_subplots(rows=plot_rows, cols=plot_cols, subplot_titles=("",""), shared_xaxes=True)
fig.add_trace(go.Bar(x=data['time_delta_with_previous_rental_in_minutes'].unique(), y=seuil_mobile, name="mobile", marker_color='royalblue'), row=1, col=1)
fig.add_trace(go.Bar(x=data['time_delta_with_previous_rental_in_minutes'].unique(), y=seuil_connect, name="connect", marker_color='cyan'), row=1, col=1)
fig.add_trace(go.Bar(x=data['time_delta_with_previous_rental_in_minutes'].unique(), y=seuil_total, name="total", marker_color='darkblue'), row=1, col=1)
fig.update_yaxes(title='Pourcentage de locations affectées')
fig.update_xaxes(title='Time Delta')
fig.add_trace(go.Scatter(x=data['time_delta_with_previous_rental_in_minutes'].unique(), y=seuil_total, mode='markers', showlegend=False, marker_color='darkblue'), row=1, col=2)
fig.add_trace(go.Scatter(x=data['time_delta_with_previous_rental_in_minutes'].unique(), y=seuil_mobile, mode='markers', showlegend=False, marker_color='royalblue'), row=1, col=2)
fig.add_trace(go.Scatter(x=data['time_delta_with_previous_rental_in_minutes'].unique(), y=seuil_connect, mode='markers', showlegend=False, marker_color='cyan'), row=1, col=2)
fig.add_trace(go.Histogram(y=seuil_mobile*10, name="mobile", nbinsx=20, marker_color='royalblue', showlegend=False), row=1, col=2)
fig.add_trace(go.Histogram(y=seuil_connect*10, name="connect", marker_color='cyan', showlegend=False), row=1, col=2)
st.plotly_chart(fig, use_container_width=True)

newnames = {'wide_variable_0':'mobile', 'wide_variable_1': 'connect', 'wide_variable_2': 'sum'}
fig = px.bar(x=data['time_delta_with_previous_rental_in_minutes'].unique(), y=[seuil_mobile, seuil_connect, seuil_total], labels=newnames, color_discrete_sequence=['cyan','royalblue', 'darkblue'])
fig.update_yaxes(title='Pourcentage de locations affectées')
fig.update_xaxes(title='Time Delta')
fig.for_each_trace(lambda t: t.update(name = newnames[t.name], legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])))
st.plotly_chart(fig, use_container_width=True)

st.markdown("""On remarque encore une fois une stabilité dans la différence entre le type de check-in. On retrouve les 8% des valeurs correspondantes aux annulations.

Nous avons précédemment créer un dataframe en excluant les outliers, cad les valeurs qui sont à +- 2 écarts types de la moyenne 
mais il nous faut vérifier certains resultats

Il existe une sorte de plateau qui est atteint plus rapidement avec le type de check-in `Connect`, entre 120 et 180 minutes. On peut donc choisir un seuil entre ces deux valeurs

En comparant le minimum et le maximum des retards en minutes sans les outliers, cela fait +- 30 heures maximum. C'est trop et nous souhaitons voir la distribution de ce dataset. Nous allons supprimer ces valeurs pour ne garder qu'un range de valeurs entre -4h et 4h""")

st.markdown("""
    ------------------------
""")

st.subheader("Distribution des retards dans un range de plus ou moins 4 heures")

data_without_outliers_range = data_without_outliers[(data_without_outliers['delay_at_checkout_in_minutes'] > -240) & (data_without_outliers['delay_at_checkout_in_minutes'] < 240)]
fig = px.histogram(data_without_outliers_range, x='delay_at_checkout_in_minutes', color='checkin_type', barmode='overlay', marginal='box', color_discrete_sequence=['cyan','royalblue'])
st.plotly_chart(fig, use_container_width=True)

st.markdown("""Plus on réduit le range des tranches horaires d'études, plus on supprime les outliers et plus le type de check-in `mobile` crée des retards sur les courses.

Testons en regardant par tranche horaire de 30 minutes la différence de retard :""")

specs = np.repeat({'type':'pie'}, 10).reshape(2, 5).tolist()
fig = make_subplots(rows=2, cols=5, specs=specs)
range_minutes = np.linspace(0, 240, 5)
colors = ['cyan','royalblue']
i,j = 0,1

for ranges in range_minutes:
    advance_connect = len(data_connect[data_connect['delay_at_checkout_in_minutes'] < ranges])
    delays_connect = len(data_connect[data_connect['delay_at_checkout_in_minutes'] >= ranges])
    advance_mobile = len(data_mobile[data_mobile['delay_at_checkout_in_minutes'] < ranges])
    delays_mobile = len(data_mobile[data_mobile['delay_at_checkout_in_minutes'] >= ranges])
  
    fig.add_trace(go.Pie(labels=['Disponible', 'Non disponible'], values=[advance_connect, delays_connect], name=f"{ranges} minutes between ck_in & check out"), j, i+1)
    fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20, marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    fig.add_trace(go.Pie(labels=['Disponible', 'Non disponible'], values=[advance_mobile, delays_mobile], name=f"{ranges} minutes between ck_in & check out"), j+1, i+1)
    fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20, marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    i+=1
fig.update_traces(hole=.4, hoverinfo="label+percent+name")
    
fig.update_layout(
    title_text="Retard entre deux courses en fonction du type de check-in",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Pas de retard', x=0.02, y=0.50, font_size=26, showarrow=False),
                 dict(text='< 60 min', x=0.245, y=0.50, font_size=26, showarrow=False),
                 dict(text='< 120 min', x=0.50, y=0.50, font_size=26, showarrow=False),
                 dict(text='< 180 min', x=0.76, y=0.50, font_size=26, showarrow=False),
                 dict(text='> 240 min', x=0.99, y=0.50, font_size=26, showarrow=False),
                 dict(text='mobile', x=-0.01, y=1, font_size=38, showarrow=False,textangle=-90),
                 dict(text='connect', x=-0.01, y=-0.05, font_size=38, showarrow=False, textangle=-90),
                ])
    
st.plotly_chart(fig, use_container_width=True)

st.markdown(""" Plus nous mettons un délais de retard important, et plus les véhicules sont disponibles. Alors que, pour la totalité des véhicules, il y a plus de retard avec les type de check-in `mobile` que `connect`,
on remarque une inversion de cette tendance : plus on prend des delais de retard important, et plus l'ecart qu'il y avait en terme de retard entre `mobile` et `connect` se réduit jusqu'a s'inverser et devenir majoritaire 
pour le type `connect`.""")

st.markdown("""
    ------------------------
""")