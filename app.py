from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return "Bienvenue sur la plateforme de prédiction de trafic réseau !"

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json  # Supposons que les données soient envoyées en JSON
#     # Dummy model logic for testing
#     input_data = np.array(data['input'])
#     prediction = dummy_model(input_data)
#     return jsonify({"result": prediction.tolist()})

# def dummy_model(input_data):
#     # Simple dummy model that returns the sum of the input data
#     return np.sum(input_data, axis=1)

if __name__ == '__main__':
    app.run(debug=True)
