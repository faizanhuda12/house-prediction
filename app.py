import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle, os

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('house_sale.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    print(prediction)
    output = "${:,.2f}".format(int(prediction))

   

    return render_template('house_sale.html', prediction_text='In 2014 the house price was:  {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
     
    output = prediction[0]
    return jsonify(output)

#if __name__ == "__main__":
#    app.run(debug=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
