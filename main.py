import pandas as pd
import numpy as np
import pickle
import joblib
from flask import Flask, render_template, request, Response, send_file, url_for 
from source import customer_churn_model
from functools import wraps

app = Flask(__name__, static_url_path='/static')
app.debug = True

@app.route('/', methods = ['GET'])
def categories():

    locations = ['France', 'Spain', 'Germany']

    genders = ['Male', 'Female']

    tenures = [0,1,2,3,4,5,6,7,8,9]

    number_prods = [1,2,3,4]

    creditcards = ['Yes', 'No']

    actives = ['Yes', 'No']


    return render_template('home.html', locations = locations, genders=genders, number_prods = number_prods, tenures=tenures, creditcards = creditcards, actives=actives)


@app.route('/result', methods = ['GET', 'POST'])
def prediction():
    
    if request.method == 'POST':
        credit_score = request.form.get('credit_score')
        location = request.form.get('location')
        gender = request.form.get('gender')
        age = request.form.get('age')
        tenure = request.form.get('tenure')
        balance = request.form.get('balance')
        number_prod = request.form.get('number_prod')
        creditcard = request.form.get('creditcard')
        active = request.form.get('active')
        salary = request.form.get('salary')

        result = customer_churn_model.chrun_classifier(int(credit_score), location, gender, float(age), 
                                                    tenure, float(balance), int(number_prod), creditcard,
                                                        active, float(salary))
        
        return render_template('result.html', result= result)

if __name__=='__main__':
    app.run(host='0.0.0.0')