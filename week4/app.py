import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# Title and description
st.title("Tennis Match Prediction Using Naive bayes")
st.write("""
This application uses a Naive Bayes classifier to predict whether a tennis match will be enjoyed based on the given weather conditions.
""")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("First 5 values of the dataset:")
    st.write(data.head())

    # Obtain Train data and Train output
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Convert categorical features to numbers
    le_sky = LabelEncoder()
    X['sky'] = le_sky.fit_transform(X['sky'])

    le_airTemp = LabelEncoder()
    X['airTemp'] = le_airTemp.fit_transform(X['airTemp'])

    le_humidity = LabelEncoder()
    X['humidity'] = le_humidity.fit_transform(X['humidity'])

    le_wind = LabelEncoder()
    X['wind'] = le_wind.fit_transform(X['wind'])

    le_water = LabelEncoder()
    X['water'] = le_water.fit_transform(X['water'])

    le_forecast = LabelEncoder()
    X['forecast'] = le_forecast.fit_transform(X['forecast'])

    le_enjoySport = LabelEncoder()
    y = le_enjoySport.fit_transform(y)

    st.write("Transformed Train data:")
    st.write(X.head())
    
    st.write("Transformed Train output:")
    st.write(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Train the classifier
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Calculate accuracy
    accuracy = accuracy_score(classifier.predict(X_test), y_test)
    st.write(f"Accuracy of the model is: {accuracy}")

    # Predict on new data (interactive part)
    st.write("Predict Enjoy Sport for new data:")
    
    sky = st.selectbox("Sky", le_sky.classes_)
    airTemp = st.selectbox("Air Temperature", le_airTemp.classes_)
    humidity = st.selectbox("Humidity", le_humidity.classes_)
    wind = st.selectbox("Wind", le_wind.classes_)
    water = st.selectbox("Water", le_water.classes_)
    forecast = st.selectbox("Forecast", le_forecast.classes_)

    if st.button("Predict"):
        new_data = pd.DataFrame([[sky, airTemp, humidity, wind, water, forecast]], 
                                columns=["sky", "airTemp", "humidity", "wind", "water", "forecast"])
        new_data['sky'] = le_sky.transform(new_data['sky'])
        new_data['airTemp'] = le_airTemp.transform(new_data['airTemp'])
        new_data['humidity'] = le_humidity.transform(new_data['humidity'])
        new_data['wind'] = le_wind.transform(new_data['wind'])
        new_data['water'] = le_water.transform(new_data['water'])
        new_data['forecast'] = le_forecast.transform(new_data['forecast'])

        prediction = classifier.predict(new_data)
        result = le_enjoySport.inverse_transform(prediction)[0]
        st.write(f"The model predicts that you will {'enjoy' if result == 'yes' else 'not enjoy'} the sport.")
