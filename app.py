# ============================= scripts ==================================
## Importation des librarys
import joblib
#import uvicorn
from flask import Flask, jsonify
import pandas as pd
import pickle
#from lightgbm import LGBMClassifier

#avoid some errors  

import warnings
warnings.filterwarnings("ignore", category = UserWarning)


#  PATH = 'dataset/'
# Chargement des données  
def load_data ():
    data = pd.read_csv('data_try.csv')
    # Un échantillon à tester
    #data = data.sample(frac=0.05)
    # Garder les colonnes avec 20 % ou plus de valeurs manquantes
    data = data.dropna(thresh=0.8*len(data), axis=1)
    return data
# Chargement du modèle  
model = joblib.load('LGBM_P7.pkl')
# Chargement du preprocessor
loaded_preprocessor = joblib.load('preprocessor_P7.joblib')

# Preprocessing
data = load_data()
X = data.drop(['SK_ID_CURR'], axis=1)
X_ = loaded_preprocessor.fit_transform(X)
X = pd.DataFrame(X_, index=X.index, columns=X.columns)
X['SK_ID_CURR'] = data['SK_ID_CURR']



#app = flask.Flask(__name__)
app = Flask(__name__)


@app.route('/')
def hello():
    return "Bienvenue, L'API pour mon projet d'Openclassrooms"


@app.route('/prediction_credit/<id_client>')  #, methods=['GET'])
def prediction_credit(id_client):

    print('id client = ', id_client)
   
    #Récupération des données du client en question
   
    ID = int(id_client)
    data_client = X[X['SK_ID_CURR'] == ID]
    data_client = data_client.drop(['SK_ID_CURR'], axis=1)
   
    print('La taille du vecteur data_client  = ', data_client.shape)

    proba = model.predict_proba(data_client)
 
    #DEBUG
    print('L''identificateur du client : ', id_client)
 
    dict_final = {
        'proba' : float(proba[0][0])
        ##  Ajouter d'autres parametres
        }
   
    print('Lancer une nouvelle Prédiction : \n', dict_final)
   

     # Sauvegarde le résultat sous forme de JSON file
       
    return jsonify(dict_final)


#  lancement de l'application   (  mode local  et non en mode production  )

#def create_app():
#       return app


if __name__ == "__main__":
    app.debug = True
    app.run()
