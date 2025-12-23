from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
import joblib
import os

def train():
    data=pd.read_csv("Iris.csv")

    x=data.drop(columns=['Id',"Species"])
    y=data['Species']

    model = RandomForestClassifier()
    model.fit(x,y)


    os.makedirs('model',exist_ok=True)
    joblib.dump(model,'model/model.pkl')

    print("trained")

if __name__ == "__main__":
    train()