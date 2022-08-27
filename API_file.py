#=============================================scripts========================
# Importation des librarys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import uvicorn
#avoid some errors  
import warnings
warnings.filterwarnings("ignore", category = UserWarning)
#=============================================

# Chargement des données  
def load_data ():
    data = pd.read_csv('C:/Users/HP ELITEBOOK/Desktop/Nouveau dossier/test_data.csv')
    # Un échantillon à tester
    #data = data.sample(frac=0.05)
    # Garder les colonnes avec 20 % ou plus de valeurs manquantes
    data = data.dropna(thresh=0.8*len(data), axis=1)
    return data
# Chargement du modèle  
model = joblib.load('C:/Users/HP ELITEBOOK/Desktop/Nouveau dossier/LGBM_P7.pkl')
# Chargement du preprocessor
loaded_preprocessor = joblib.load('C:/Users/HP ELITEBOOK/Desktop/Nouveau dossier/preprocessor_P7.joblib')

# Preprocessing
data = load_data()
X = data.drop(['SK_ID_CURR'], axis=1)
X_ = loaded_preprocessor.fit_transform(X)
X = pd.DataFrame(X_, index=X.index, columns=X.columns)
X['SK_ID_CURR'] = data['SK_ID_CURR']
#=============================================

#create the application
app = FastAPI(
    title = "Credit Score API",
    version = 1.0,
    description = "Simple API to make predict cluster of client."
)

@app.get("/")
def index():
    return {"message":"Évaluez la capacité de crédit de votre client"}


# Définir une classe contenant les ID à saisir:
class Customer(BaseModel):
    id: int


@app.post("/",tags = ["credit_score"])
async def get_prediction(client_id: Customer):

    if client_id.dict()['id'] not in X['SK_ID_CURR'].unique():
        raise HTTPException(
            status_code=404, detail=f"Client ID {client_id.dict()['id']} not found")
        
    #df = X.loc[data['SK_ID_CURR'] == client_id['id']] #client_id
    df = X[X['SK_ID_CURR'] == int(client_id.dict()['id'])]
    df = df.drop(['SK_ID_CURR'], axis=1)
    per_pos = model.predict_proba(df)[0][1]

    #return JSONResponse({"Credit score":round(per_pos, 3)})
    return {
    'Credit score': round(per_pos, 3)
    }
#=============================================

#  lancement de l'application   (  mode local  et non en mode production  )
#  affichage:   Located at: http://127.0.0.1:8000/AnyNameHere

#import uvicorn
#import nest_asyncio

#nest_asyncio.apply()
#uvicorn.run(app, host="127.0.0.1",port=8000)
#=============================================

#if __name__ == "__main__":
#    app.debug = True
#    app.run()
