from fastapi import APIRouter
from api.wine import Wine

router = APIRouter()

#realise the best wine predict
@router.post("/api/predict")
async def get_prediciton():
    # TODO: use fonction of your model to predict the best wine
    # and return 
    wine = {}
    return {"The best wine predict": wine}
    
# generate combinaison of data to give the "perfect" wine
@router.get("/api/predict")
async def generate_combinaison(id: int,fixed_acidity: float, volatile_acidity: float, citric_acid: float,
                        residual_sugar: float, chlorides: float, free_sulfur_dioxide: float,
                        total_sulfur_dioxide: float, density: float, pH: float, sulphates: float,
                        alcohol: float, quality: float):
    
    # TODO: call fonction in the model with selections of data and give the note  
    wine_note = 0            
    return {"The model predict the note of the wine": wine_note}

