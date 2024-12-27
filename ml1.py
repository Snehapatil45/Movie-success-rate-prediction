import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess the dataset
data = pd.read_csv('movie_success_rate.csv')
data = data.dropna()

# One-hot encode the Genre column
genres = data['Genre'].str.get_dummies(sep=',')
data = pd.concat([data, genres], axis=1)
data = data.drop(columns=['Title', 'Description', 'Director', 'Actors', 'Genre'])

# Ensure the training and prediction feature names match exactly
all_features = data.columns.drop('Success')
X = data[all_features]
y = data['Success']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
svm_model = SVC(kernel='rbf', random_state=42).fit(X_scaled, y)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_scaled, y)

# Create a template DataFrame with the correct feature names and order
template_df = pd.DataFrame(columns=all_features)

st.title('Movie Success Predictor')

st.header('Enter Movie Details')
rating = st.number_input('Rating', min_value=0.0, max_value=10.0, step=0.1)
votes = st.number_input('Votes', min_value=0)
revenue = st.number_input('Revenue (Millions)', min_value=0.0, step=0.1)
metascore = st.number_input('Metascore', min_value=0.0, max_value=100.0, step=1.0)

# Genre-related features based on the unique genres in the dataset
genre_features = genres.columns.tolist()
genre_values = {genre: st.checkbox(genre, value=False) for genre in genre_features}

if st.button('Predict'):
    # Fill the template DataFrame with user input
    user_input = template_df.copy()
    user_input.loc[0, ['Rating', 'Votes', 'Revenue (Millions)', 'Metascore']] = [rating, votes, revenue, metascore]

    for genre in genre_features:
        user_input.loc[0, genre] = int(genre_values[genre])
    
    user_input = user_input.fillna(0)  # Fill other columns with zeros
    
    # Ensure the input DataFrame matches the training features
    user_input = user_input[all_features]
    
    # Scale the user input
    user_input_scaled = scaler.transform(user_input)
    
    # Make predictions with both models
    svm_prediction = svm_model.predict(user_input_scaled)[0]
    rf_prediction = rf_model.predict(user_input_scaled)[0]

    st.write("## Predictions")
    st.write(f"SVM Prediction: {'Success' if svm_prediction == 1 else 'Fail'}")
    st.write(f"Random Forest Prediction: {'Success' if rf_prediction == 1 else 'Fail'}")
