import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def Diabetes(inputeg):
    # Load the dataset
    dataset = pd.read_csv('model/dataset.csv')

    # Retrain the model with only 'HbA1c_level' as a feature
    # Define features and target variable
    X = dataset[['HbA1c_level']]
    y = dataset['diabetes']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating and training the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Making predictions
    y_pred = model.predict(X_test)

    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy)

    # Function to classify HbA1c level
    def classify_HbA1c_level(HbA1c):
        if HbA1c >= 5.6 and HbA1c <= 6.4:
            result = "Type 1 Diabetes"
        elif HbA1c > 6.4:
            result = "Type 2 Diabetes"
        else:
            result = "No diabetes"
        print(result)
        return result

    # Assuming 'HbA1c_level' is the only feature for prediction
    new_data = [[float(inputeg)]]
    prediction = model.predict(new_data)

    # Directly use the classify_HbA1c_level function for output
    prediction_class = classify_HbA1c_level(float(inputeg))

    # Visualize HbA1c level
    HbA1c_value = float(inputeg)
    fig, ax = plt.subplots()
    ax.bar(['HbA1c Level'], [HbA1c_value], color='blue')
    ax.set_ylim(0, max(10, HbA1c_value + 2))
    ax.set_ylabel('HbA1c Level')
    ax.set_title(f'Predicted: {prediction_class}')
    plt.axhline(y=5.6, color='r', linestyle='--', label='Pre-diabetes threshold')
    plt.axhline(y=6.4, color='r', linestyle='--', label='Diabetes threshold')
    plt.legend()
    plt.show()

    return prediction_class

# Example usage
if __name__ == "__main__":
    input_eg = 7.0  # Example input, replace with actual input
    result = Diabetes(input_eg)
    print(f'Diabetes Prediction: {result}')
