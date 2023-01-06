from fastapi import FastAPI, Request
import uvicorn
from pydantic import BaseModel, validator
from typing import Union
import joblib
import pickle
import json
import pandas as pd 

app = FastAPI(
    title='GetAround API',
    description="""Voici l'API du projet Get Around pour prédire le prix d'un véhicule de location.

Cette API est basé sur le framework FastAPI et contient un dataset de 10k lignes et 9 colonnes :</br>
* `model_key` : Marque du véhicule emprunté</br>
* `mileage` : Nombre de kms</br>
* `engine_power` : Puissance du moteur</br>
* `fuel` : type de carburant</br>
* `paint_color` : couleur du véhicule</br>
* `car_type` : type de véhicule (sport, Van, coupé,...)</br>
* `private_parking_available` : si parking privé disponible (Oui ou Non)</br>
* `has_gps` : possède un GPS (Oui ou Non)</br>
* `has_air_conditioning` : possède un climatiseur (Oui ou Non)</br>
* `automatic_car` : le véhicule est il automatique (Oui ou Non)</br>
* `has_getaround_connect` : possède l'option GetAround Connect (permet un check-in sans rencontre entre proprio / locataire) (Oui ou Non)</br>
* `has_speed_regulator` : possède un régulateur de vitesse (Oui ou Non)</br>
* `winter_tires` : possède des pneus d'hiver (pour la neige) (Oui ou Non)</br>
* `rental_price_per_day` : prix de la location pour la journée (Target)</br>
    
Dans cette API, on peut trouver 6 endpoints:</br>
    - **/preview**: renvoie une overview du dataset (sous forme de dictionnaire)</br>
    - **/predict**: retourne le prix prédit d'une voiture</br>
    - **/unique-values**: renvoie les valeurs uniques d'une colonne (sous forme de liste)</br>
    - **/groupby**: renvoie les données groupées d'une colonne (sous forme de dictionnaire)</br>
    - **/filter-by**: renvoie les données filtrées d'une colonne (sous forme de dictionnaire)</br>
    - **/quantile**: renvoie le quantile d'une colonne (sous forme de flottant ou de chaîne)"""
)

@app.get("/")
async def root():
    message = """Bienvenue dans l'API Getaround. Ajoutez /docs à cette adresse pour voir la documentation de l'API sur le dataset contenant les prix des locations"""
    return message

# On definit une classe avec toutes les features pour faire les endpoints sur les predictions
class Features(BaseModel):
    model_key: str
    mileage: Union[int, float]
    engine_power: Union[int, float]
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

# Pour toutes les colonnes (exceptées les Bools) on crée des fonctions qui vont tester si les valeurs en input sont correctes
    @validator('model_key')
    def model_key_is_valid(cls, v):
        assert v in ['Citroën', 'Peugeot', 'PGO', 'Renault', 'Audi', 'BMW', 'Ford',
       'Mercedes', 'Opel', 'Porsche', 'Volkswagen', 'KIA Motors','Alfa Romeo', 'Ferrari', 'Fiat', 'Lamborghini', 'Maserati',
       'Lexus', 'Honda', 'Mazda', 'Mini', 'Mitsubishi', 'Nissan', 'SEAT','Subaru', 'Toyota', 'Suzuki', 'Yamaha'], \
        f"model_key doit être une des valeurs de cette liste: ['Citroën', 'Peugeot', 'PGO', 'Renault', 'Audi', 'BMW', 'Ford', \
       'Mercedes', 'Opel', 'Porsche', 'Volkswagen', 'KIA Motors','Alfa Romeo', 'Ferrari', 'Fiat', 'Lamborghini', 'Maserati', \
       'Lexus', 'Honda', 'Mazda', 'Mini', 'Mitsubishi', 'Nissan', 'SEAT','Subaru', 'Toyota', 'Suzuki', 'Yamaha']"
        return v

    @validator('fuel')
    def fuel_is_valid(cls, v):
        assert v in ['diesel', 'petrol', 'hybrid_petrol', 'electro'], \
        f"fuel doit être une des valeurs de cette liste: ['diesel', 'petrol', 'hybrid_petrol', 'electro']"
        return v
    
    @validator('paint_color')
    def paint_color_is_valid(cls, v):
        assert v in ['black', 'white', 'red', 'silver', 'grey', 'blue', 'orange','beige', 'brown', 'green'], \
        f"paint_color doit être une des valeurs de cette liste: ['black', 'white', 'red', 'silver', 'grey', 'blue', 'orange','beige', 'brown', 'green']"
        return v
    
    @validator('car_type')
    def car_type_is_valid(cls, v):
        assert v in ['sedan', 'hatchback', 'suv', 'van', 'estate', 'convertible', 'coupe', 'subcompact'], \
        f"car_type doit être une des valeurs de cette liste: ['sedan', 'hatchback', 'suv', 'van', 'estate', 'convertible', 'coupe', 'subcompact']"
        return v

    @validator('mileage')
    def mileage_is_positive(cls, v):
        assert v >= 0, f"mileage doit être positif"
        return v
    
    @validator('engine_power')
    def engine_power_is_positive(cls, v):
        assert v >= 0, f"engine_power doit être positif"
        return v


# endpoint de la prediction du prix d'une voiture
@app.post("/predict")
async def predict(features:Features):
    """Prediction du prix d'une voiture. 
Exemple de données d'entrée:
{
  "model_key": "Citroën",
  "mileage": 140411,
  "engine_power": 100,
  "fuel": "diesel",
  "paint_color": "black",
  "car_type": "convertible",
  "private_parking_available": true,
  "has_gps": true,
  "has_air_conditioning": false,
  "automatic_car": false,
  "has_getaround_connect": true,
  "has_speed_regulator": true,
  "winter_tires": true
}
Devrait retourner : "prediction": 108.75097498268137

Toutes les entrées sont sensibles à la casse. 
La liste des valeurs possibles pour les colonnes catégorielles est disponible dans le endpoint /unique-values. 
Des valeurs erronées renverront un message d'erreur spécifique."""

    # Creation d'un dictionnaire contenant les features et la valeur associée du sample 
    features = dict(features)
    # Création du dataframe contenant les features
    sample_df = pd.DataFrame(columns=['model_key', 'mileage', 'engine_power', 'fuel', 'paint_color','car_type', 'private_parking_available', 'has_gps',
       'has_air_conditioning', 'automatic_car', 'has_getaround_connect','has_speed_regulator', 'winter_tires'])
    # Ajout des valeurs du sample au dataframe sample_df
    sample_df.loc[0] = list(features.values())
    # Chargement du modèle avec pickle
    with open('../../src/models/gradient_boosting_model', 'rb') as f:
        gbr_model = pickle.load(f)
    # Chargement du preprocessor avec joblib
    preprocessor = joblib.load('../../src/models/preprocessor.pkl')
    # Application du preprocessing sur sample_df
    X = preprocessor.transform(sample_df)
    # On fait les predictions avec le modèle Gradient Boosting Regressor
    predictions = gbr_model.predict(X)
    # On observe la prediction faites
    return {"prediction" : predictions[0]}


# Endpoints to explore the dataset

@app.get("/preview")
async def preview(rows: int):
    """ Donne une preview du dataset : on doit entrer le nombre de lignes sous forme d'integer"""
    df = pd.read_csv('../../src/get_around_pricing_project.csv')
    preview = df.head(rows)
    return preview.to_dict()

@app.get("/unique-values")
async def get_unique(column: str):
    """Valeurs uniques pour une colonne donnée : nom de la feature (sous forme de string). 
    Exemple de suffixe : /unique-values?column=model_key
    Ne fonctionne que pour des valeurs catégorielles (`Rental_price_per_day` ou `Mileage` renverra une erreur)."""
    df = pd.read_csv('../../src/get_around_pricing_project.csv')
    select = df[column].unique()
    return list(select)

@app.get("/groupby")
async def groupby_agg(column:str,parameter:str):
    """ Récupère les données regroupées pour une colonne donnée :
    Les paramètres d'entrée sont:
    1. la colonne (sous forme de string),
    2. le paramètre d'agrégation (sous forme de string).
    Exemple de suffixe : /groupby?column=model_key&parameter=mean"""
    df = pd.read_csv('../../src/get_around_pricing_project.csv')
    df_groupby = df.groupby(column).agg(parameter)
    return df_groupby.to_dict()

@app.get("/filter-by")
async def get_filtered(column:str,category:str):
    """ Obtenir les données filtrées pour une colonne donnée : 
    Les paramètres d'entrée sont:
    1. colonne (sous forme de string),
    2. catégorie (sous forme de string).
    Exemple de suffixe : /filter-by?column=model_key&category=Toyota"""
    df = pd.read_csv('../../src/get_around_pricing_project.csv')
    filtered_df = df.loc[data[column] == category]
    return filtered_df.to_dict()

@app.get("/quantile")
async def get_quantile(column:str,decimal:float):
    """Obtenir les quantiles pour une colonne donnée : 
    Les paramètres d'entrée sont:
    1. colonne (sous forme de string),
    2. quantile (flottant entre 0 et 1, ex : 0,75).
    La méthode d'interpolation utilisée lorsque le quantile souhaité est compris entre 2 points de données est 'le plus proche'(nearest) pour les données catégorielles et 'linéaire'(linear) pour les données numériques.
    Exemple de suffixe : /quantile?column=mileage&decimal=0.25"""
    df = pd.read_csv('../../src/get_around_pricing_project.csv')
    try:
        quantile_df = df[column].quantile(decimal,interpolation='linear')
    except:
        quantile_df = df[column].quantile(decimal,interpolation='nearest')
    return quantile_df


if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 4000, debug=True, reload=True)
