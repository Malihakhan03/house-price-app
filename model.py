import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model(file):
    # Load dataset
    data = pd.read_csv(file)

    # Assume dataset has these columns
    X = data[['area', 'bedrooms', 'bathrooms']]
    y = data['price']

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    return model