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

st.sidebar.write("Dashboard made by [@DavidT](https://github.com/DavidTGAUTIER)")

st.header("Analyse des prix des locations")

st.subheader("A la recherche du seuil optimal pour optimiser la rentabilité des courses")

st.markdown("""Pour améliorer l'expérience de l'utilisateur, nous devons répondre à ces questions :

* Combien d'argent perd un propriétaire de véhicule à cause des retards ?
* Comment quantifier le ratio risques/bénéfices ?
* Quel seuil devrions nous utiliser pour améliorer le ratio risques/bénéfices ?

Nous allons prendre une durée de location égale à 24h.

Nous avons le choix de prendre la median ou la moyenne pour connaitre le prix (moyen ou median) d'une course. Regardons la distribution de cette variable""")

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
    # permet de memorizer les executions de fonctions (cad les mettre en cache pour eviter d'avoir besoin de les relancer : on choisit une valeur de 900s, cad que si pendant 15min il y a un changement, charge auto la fonction)
    @st.experimental_memo(ttl=600)
    def import_data(bucket_name, file_path):
        bucket = client.bucket(bucket_name)
        content = bucket.blob(file_path).download_as_string().decode('utf-8')
        return content

    bucket_name = "get_around_data"
    file_path_pricing = "pricing_cleaned.csv"
    file_path_delays = "delays_cleaned.csv"

    data_pricing = import_data(bucket_name, file_path_pricing)
    data_delays = import_data(bucket_name, file_path_delays)
    data_pricing = pd.read_csv(io.StringIO(data_pricing))
    data_delays = pd.read_csv(io.StringIO(data_delays))


if local:
    # Importation des données depuis environnement local
    @st.cache(allow_output_mutation=True)
    def import_data(path):
        data = pd.read_csv(path)
        return data

    path_pricing = './src/pricing_cleaned.csv'
    path_delays = './src/delays_cleaned.csv'
    path_no_outliers_pricing = './src/pricing_without_outliers.csv'
    path_no_outliers_delays ='./src/delays_without_outliers.csv'
    data_pricing = import_data(path_pricing)
    data_delays = import_data(path_delays)
    pricing_without_outliers = import_data(path_no_outliers_pricing)
    delays_without_outliers = import_data(path_no_outliers_delays)

data_load_state.text("Données disponibles")

st.markdown("""
    ------------------------
""")

st.subheader("Analyse de fond sur la determination d'un seuil de rentabilité")

# on ne prend pas les outliers pour avoir un nombre total de sample plus exacte (nous ne connaissons pas les raisons des NaN)
delays_without_nan = data_delays.dropna(subset=['delay_at_checkout_in_minutes'])

col1, col2= st.columns(2)

with col1:
    st.markdown(""" Distribution du prix d'une location """)
    fig1 = px.histogram(data_pricing['rental_price_per_day'], color_discrete_sequence=['cyan'])
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown(""" Distribution du temps de retards(sans outliers) """)
    threshold_range_delais = delays_without_nan[(delays_without_nan['delay_at_checkout_in_minutes']>-1500) & (delays_without_nan['delay_at_checkout_in_minutes']<1500)]
    fig2 = px.histogram(threshold_range_delais['delay_at_checkout_in_minutes'], color_discrete_sequence=['royalblue'])
    st.plotly_chart(fig2, use_container_width=True)
   
st.markdown("""Comme nous sommes sur une distribution gaussienne, la moyenne et la median sont equivalente donc nous prendrons la moyenne.

Pour connaitre le cout total des annulations, nous pouvons prendre le prix moyen d'une course multiplié par la totalité des courses annulées : on obtient une perte totale de revenu pour l'ensemble des locations de 395765$""")

st.markdown("""Notre objectif est d'optimiser les revenus des propriétaires. Nous devons assumer le fait que toutes les annulations de location sont dues à des retards et qu'aucun propriétaire ne gagne de l'argent après l'heure de départ prévue. De plus, si il y  a le moindre retard, un utilisateur va annuler sa course
Par la suite, nous ne prendrons que les valeurs qui ne sont pas des `NaN`.

Pour un départ tardif, 9404 courses ont coutées 159737$ si nous nous basons sur un tarif à la minute. Nous pouvons donc diminuer la perte totale des revenus en faisant payer le temps supplémentaire par un tarif à la minute dés qu'il y a un retard.
Pour cela, nous allons retrancher le prix qu'aurait rapporté les retards par rapport à la perte totale liées aux retards""")

st.markdown("""Les pertes s'élèvent à un montant de 236028$ pour une durée de 24h. Nous obtenons un montant de pertes liées aux retards par heure donc pour connaitre la valeur de ce montant par jour, nous devons multiplier par 24 le montant des pertes totales, qui va ensuite être divisé par le montant des pertes réduites. 

Si les locations annulées duraient moins de 9 heures et 30 minutes, les revenus supplémentaires provenant des départs tardifs et la perte des locations annulées atteignent un certain seuil de rentabilité.""")

st.subheader("Calcul du seuil de rentabilité")

st.markdown("""En multipliant la somme des retards d'une certaine tranche horaire obtenus par le prix moyen d'une course, on peut connaitre le montant total des retards de cette tranche horaire.
Nous allons pouvoir afficher sur un graphique les pertes et leur evolution au cours de 24h""")

h = 24
prix_moyen_course = data_pricing['rental_price_per_day'].mean()
prix_moyen_course_par_heure = prix_moyen_course / (24*60)

annulations = (delays_without_nan['state'] == 'canceled').sum()
pertes_totales = prix_moyen_course * annulations

only_delays = delays_without_nan[delays_without_nan['delay_at_checkout_in_minutes'] > 0]
total_retards = only_delays.loc[:, 'delay_at_checkout_in_minutes'].sum()
revenu_retard = int(total_retards * prix_moyen_course_par_heure)

heures_rentable = (revenu_retard / pertes_totales) * 24
total_revenus_retard = []

interval_range = np.arange(0, 60*24, step=30)
for interval in interval_range:
    threshold = delays_without_nan[delays_without_nan['delay_at_checkout_in_minutes'] > interval]
    somme_retard = threshold.loc[:, 'delay_at_checkout_in_minutes'].sum()
    somme_revenus_retard = somme_retard * prix_moyen_course_par_heure
    total_revenus_retard.append(somme_revenus_retard)
    
total_revenus_retard.reverse()
h = 24
fig = make_subplots(rows = 1, cols = 2, subplot_titles = ("Cout de revient des annulations sur 24h", "Distribution des annulations"))
fig.add_trace(go.Scatter(x=interval_range/60, y=total_revenus_retard, marker_color='cyan', fill='tozeroy'), row=1, col=1)
fig.add_hline(y=int(pertes_totales/h*heures_rentable))
fig.add_trace(go.Histogram(x=total_revenus_retard, marker_color='royalblue'),row=1, col=2)
fig.update_layout(title = go.layout.Title(text = "Distribution des retards", x=0.5), showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""Il faudrait connaitre la **durée de chaque trajet**, **le temps qu'une voiture reste inutilisée** ou bien si **il existe d'autres voitures disponibles** ect..., pour estimer avec précision les pertes.

De plus, nous admettrons que:

* Chaque minute de retard entraîne une annulation
* Chaque location ont déja une prochaine location de prévue 
* Toutes les locations annulées auraient été une location de 24 heures

Nous allons donc calculer un taux de risque : En calculant le revenu moyen qui depend du nombre total de courses finies, du prix moyen d'une course et le prix total qu'aurait pu rapporter les retards, on se rend compte
qu'il y a une perte de 1139901$ liée au retard, qui correspond à 0.49 fois le revenu total estimé des locations qui est un montant de 2347053$.

Cela représente 50% de pertes seches que nous pouvons tenter de réduire en optimisant les délais de location. Cependant, nous devons connaitre les raisons qui poussent les utilisateurs à annuler leur course.
Nous allons fixer un seuil de risque qui est un compromis entre eviter de perdre de l'argent par des annulations et eviter que ce délais soit trop long pour maximiser les profits des courses. Nous allons fixer une pénalité pour les retards ainsi augmenter le tarif de location.
On va donc créer une fonction qui va determiner un seuil de risque en fonction des retards""")

def create_risque_threshold(data_delays, data_pricing, penalty, range_minute=30):
    
    interval_range = np.arange(0, 60*24, step=range_minute)
    taux_risque, total_revenus_retard = [], []
    prix_moyen_course = data_pricing['rental_price_per_day'].mean()
    prix_minute = prix_moyen_course / (24*60)
   
    for interval in interval_range:
        threshold = data_delays[data_delays['delay_at_checkout_in_minutes'] > interval]
        nombre_retard = threshold['delay_at_checkout_in_minutes'].count()
        somme_retard = threshold['delay_at_checkout_in_minutes'].sum()
        somme_revenus_retard = somme_retard * prix_minute
        somme_revenus_retard_penalite = somme_revenus_retard * penalty
        niveau_risque = nombre_retard * prix_moyen_course
        taux_risque.append(niveau_risque / somme_revenus_retard_penalite)
        total_revenus_retard.append(somme_revenus_retard_penalite)
        
    # On va pouvoir afficher le graphique qui représente l'evolution du risque en fonction du temps passé
    fig = make_subplots(rows = 1, cols = 2, subplot_titles = ("Seuil de risque des annulations sur 24h", "Distribution des revenus en fonction du risque"))
    h = 24
    fig.add_trace(go.Scatter(x=interval_range, y=taux_risque, marker_color='cyan', fill='tozeroy'), row=1, col=1)
    fig.add_trace(go.Histogram(x=total_revenus_retard, marker_color='royalblue'),row=1, col=2)
    fig.update_layout(title = go.layout.Title(text = "Calcul de risque", x=0.5), showlegend=False)
    return fig

st.markdown("Choisir un type de statut ")
s= st.selectbox("Selectionnez un type de statut", ['Avec outliers', 'Sans outliers'])
if s == 'Avec outliers':
    st.markdown(""" Seuil de risque avec `outliers` """)
    pricing, delays = data_pricing, data_delays
else:
    st.markdown(""" Seuil de risque sans `outliers` """)
    pricing, delays = pricing_without_outliers, delays_without_outliers
fig = create_risque_threshold(delays, pricing, penalty=2, range_minute=30)
st.plotly_chart(fig, use_container_width=True)


st.markdown("""Avec les graphiques précedents, on remaque un taux de risque de 350% qui décroit jusqu'a une valeur normal(100%) situé aux alentours de 150 minutes : nous devrions donc attendre 150 minutes entre deux locations pour baisser le nombre d'annulation.

Il faut savoir que notre étude à été effectuée sur une base de 24h (pour simplifier le process), il faudrait peut être l'étudier sur des bases plus courtes (12h ou 6h) pour comparer les résultats.

Il faudrait également avoir comme données les heures de début et de fin de toutes les locations pour se faire une idée sur les moments propices aux annulations (ce qui augmenterait ou diminuerait le taux de risque sur ces tranches horaires)""")