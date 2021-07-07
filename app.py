import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('rainfall.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('prediction.html')
    
@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 
                       'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 
                       'RainToday']
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)[0]
        
    if output == 0.0:
        res_val = " rain "
    else:
        res_val = "not rain"
        

    #return render_template('prediction.html', prediction_text = output)
    return render_template('prediction.html', prediction_text='Tomorrow it will {}'.format(res_val))

if __name__ == "__main__":
    app.run(debug=True)
    
