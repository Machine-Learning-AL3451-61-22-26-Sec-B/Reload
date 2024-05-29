import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Title and description
st.title("Iris Flower Prediction using K-Nearest Neighbors")
st.write("""
This application uses a K-Nearest Neighbors classifier to predict the species of an iris flower based on its features.
""")

# Load Iris dataset
dataset = load_iris()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(dataset["data"], dataset["target"], random_state=0)

# Train the classifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(X_train, y_train)

# Display the accuracy
accuracy = kn.score(X_test, y_test)
st.write(f"Accuracy of the model is: {accuracy:.2f}")

# Display predictions for the test set
st.write("Predictions for the test set:")

predictions = kn.predict(X_test)

# Create a dataframe to show the actual and predicted values
results = {
    "Sepal Length": X_test[:, 0],
    "Sepal Width": X_test[:, 1],
    "Petal Length": X_test[:, 2],
    "Petal Width": X_test[:, 3],
    "Actual Species": [dataset["target_names"][i] for i in y_test],
    "Predicted Species": [dataset["target_names"][i] for i in predictions]
}

results_df = pd.DataFrame(results)
st.dataframe(results_df)

# Highlight wrong predictions
wrong_predictions = results_df[results_df["Actual Species"] != results_df["Predicted Species"]]
st.write("Wrong Predictions:")
st.dataframe(wrong_predictions)

# Option to predict on new data (interactive part)
st.write("Predict the species of a new Iris flower:")
sepal_length = st.slider("Sepal length (cm)", float(dataset.data[:, 0].min()), float(dataset.data[:, 0].max()), float(dataset.data[:, 0].mean()))
sepal_width = st.slider("Sepal width (cm)", float(dataset.data[:, 1].min()), float(dataset.data[:, 1].max()), float(dataset.data[:, 1].mean()))
petal_length = st.slider("Petal length (cm)", float(dataset.data[:, 2].min()), float(dataset.data[:, 2].max()), float(dataset.data[:, 2].mean()))
petal_width = st.slider("Petal width (cm)", float(dataset.data[:, 3].min()), float(dataset.data[:, 3].max()), float(dataset.data[:, 3].mean()))

if st.button("Predict"):
    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = kn.predict(new_data)
    predicted_species = dataset["target_names"][prediction][0]
    st.write(f"The model predicts the species of the iris flower as: {predicted_species}")
