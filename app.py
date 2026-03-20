from flask import Flask, request, render_template
import numpy as np
from model import train_model

app = Flask(__name__)

model = None

@app.route('/')
def home():
    return render_template("index.html")

# Upload dataset and train
@app.route('/upload', methods=['POST'])
def upload():
    global model
    file = request.files['file']
    model = train_model(file)
    return render_template("index.html", message="Model Trained Successfully!")

# Predict
@app.route('/predict', methods=['POST'])
def predict():
    global model

    if model is None:
        return render_template("index.html", message="Upload dataset first!")

    area = float(request.form['area'])
    bedrooms = float(request.form['bedrooms'])
    bathrooms = float(request.form['bathrooms'])

    features = np.array([[area, bedrooms, bathrooms]])
    prediction = model.predict(features)

    return render_template("index.html",
                           prediction_text=f"Predicted Price: ₹ {round(prediction[0],2)}")

if __name__ == "__main__":
    app.run(debug=True)