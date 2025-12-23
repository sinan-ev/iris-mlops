from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

import joblib
import pandas as pd

def monitoring():
    refrence=pd.read_csv('Iris.csv').drop(columns=['Id','Species'])
    model= joblib.load('model/model.pkl')

    current_x=refrence.sample(5)
    current_y=model.predict(current_x)
    current= current_x.copy()
    report=Report(metrics=[DataDriftPreset(drift_share=0.3)])
    report.run(reference_data=refrence,current_data=current)

    report.save_html('model/model-report.html')
    


if __name__ == "__main__":
    monitoring()
