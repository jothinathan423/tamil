import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.metrics import PredictionErrorDisplay

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

dataset = pd.read_csv('cropdata.csv')

0
datasetunique = pd.read_csv('uniquecropdata.csv')



dataset_X = dataset.iloc[:,[0,1,2,3,4,5,6]].values
dataset_Y = dataset.iloc[:,7].values


dataset_X


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)



dataset_scaled = pd.DataFrame(dataset_scaled)




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    n=int(request.form['N'])
    p =int(request.form['P'])
    k = int(request.form['K'])
    t=float(request.form['Temperature'])
    h=float(request.form['Humidity'])
    ph=float(request.form['PH'])
    r=float(request.form['Rainfall'])
    data = np.array([[n,p,k,t,h,ph,r]])

    prediction = model.predict( sc.transform(data) )
    prediction_str = prediction[0]
    print(prediction_str)



    prediction_str =  "You can use " + prediction_str + " Crop"+"<br><br>Nitrogen level - "+str(n) + "<br> Phosphorus level - " + str(p) + "<br> Pottasium level - " + str(k) + "<br> Temperature - " + str(t) + "<br> Humidity - " + str(h) + "<br> PH - " + str(ph) + "<br> Rainfall - " + str(r)

    output = prediction_str
    print(output)

    return render_template('index.html', prediction_text=output)

@app.route('/search', methods=['POST'])
def search():
    label = request.form['cropselect']
    crop_data = datasetunique[datasetunique['label'] == label]
    if not crop_data.empty:
        crop_data = crop_data.iloc[:, :7]
        crop_data.reset_index(drop=True, inplace=True)
        rows = []
        for _   , row in crop_data.iterrows():
            rows.append([value for value in row])
            print(rows)
        return render_template('index.html', rows=rows)
    else:
        return "Label not found in the dataset"


if __name__ == "__main__":
    app.run(debug=True)
