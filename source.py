import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import joblib
import pickle


class customer_churn_model():

    def chrun_classifier(credit_score, geography, gender, age, tenure, balance, number_prod, has_credit_Card,
                         is_active, salary):
        
        data = {'CreditScore':int(credit_score), 'Geography':geography, 'Gender':gender, 'Age':int(age), 
                'Tenure': str(tenure), 'Balance':float(balance),
       'NumOfProducts': str(number_prod), 'HasCrCard': has_credit_Card, 'IsActiveMember': is_active, 
       'EstimatedSalary':float(salary)}
        
        map_encode = {'Yes':'1', 'No':'0'}
        
        df = pd.DataFrame(data, index=[0])

        df['HasCrCard'] = df['HasCrCard'].replace(map_encode)

        df['IsActiveMember'] = df['IsActiveMember'].replace(map_encode)

        preocessor = joblib.load('preprocessor.pkl')
        
        df_scaled = preocessor.transform(df)


        with open('customer_churn_catboost_model.pkl', 'rb') as pickle_file:
            model = pickle.load(pickle_file)
        
        result = model.predict_proba(df_scaled)

        return round(result[:, 1][0]*100, 2)

        