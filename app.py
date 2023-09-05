import numpy as np
from flask import Flask, request, render_template
import pickle

# create an app object using Flask 
app  = Flask(__name__)

# load the trained ML model
model = pickle.load(open('models\model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

# GET: A message is sent, an dthe server (backend) returns the data
# POST: Used to send HTML form data to the server (backend)
@app.route('/predict', methods=['POST'])
def predict():

    # fetch values from request.form.values() as an array
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text= 'Percent with heart disease is {}'.format(output))

# this line executes the file and provides the link to deployment server
if __name__ == "__main__":
    app.run()