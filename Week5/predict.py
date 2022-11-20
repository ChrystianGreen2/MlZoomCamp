from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open('model1.bin', 'rb'))
dv = pickle.load(open('dv.bin', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    X = dv.transform([data])
    y_pred = model.predict_proba(X)[0, 1]
    return jsonify(y_pred)