from pyexpat import model
from fastapi import FastAPI
from api.endpoints import model_predict, model


app = FastAPI()
app.include_router(model_predict.router)
app.include_router(model.router)


@app.get("/")
async def home():
    return {"Hello": "You can use : '/docs' at the end of URL if you want"}

