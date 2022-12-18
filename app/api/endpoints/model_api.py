from fastapi import APIRouter
import os 
from model.models import models


router = APIRouter()

#model = models.trainModel('../../datasource/Wines.csv')

# serializ a model
@router.get("/api/model")
async def get_model_serialize():
    response = "model already serialized"
    if not os.path.isfile("app/endpoints/model/rf_classifier.pkl"):
        models.serealizer()
        response = "serialized model"
    return {"message": response}

# get information about the model
@router.get("/api/model/description")
async def read_model_description():
    # TODO: Use your fonction to get the model description
    description = ""
    return {"message": description}

@router.put("/api/model")
async def add_item_model(id: int,fixed_acidity: float, volatile_acidity: float, citric_acid: float,
                        residual_sugar: float, chlorides: float, free_sulfur_dioxide: float,
                        total_sulfur_dioxide: float, density: float, pH: float, sulphates: float,
                        alcohol: float, quality: float):
    
    # new wine with the data put by the user
    new_wine = {
        "fixed_acidity" : fixed_acidity,
        "volatile_acidity" : volatile_acidity,
        "citric_acid" : citric_acid,
        "residual_sugar" : float,
        "chlorides" : residual_sugar,
        "free_sulfur_dioxide" : free_sulfur_dioxide,
        "total_sulfur_dioxide" : total_sulfur_dioxide,
        "density" : density,
        "pH" : pH,
        "sulphates" : sulphates,
        "alcohol" : alcohol,
        "quality" : quality,
        "id" : id
    } 
    # TODO: use fonction to add wine in the model 
    return {"message": "Adding an additional wine in the model"}

# retrain the model with the new parameters add
@router.post("/api/model/retrain")
async def retrain_model():
    # TODO: use fonction to retrain the model
    return {"message": "The model was retrain"}
