


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data = pd.read_csv("D:\creditcard.csv")

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
balanced_data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = balanced_data.drop(columns="Class", axis=1)
y = balanced_data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

# Display accuracy in the console for reference
print(f"Training Accuracy: {train_acc}")
print(f"Testing Accuracy: {test_acc}")


# Function to validate login
def validate_login(username, password):
    # In a real-world application, use a secure method to validate user credentials
    return username == "admin" and password == "password"



import streamlit as st
import numpy as np


# Mock function for login validation (replace with your actual validation logic)
def validate_login(username, password):
    # Example: Check if username and password match a predefined list
    valid_users = {"user1": "password1", "user2": "password2"}
    return username in valid_users and valid_users[username] == password


# Home Page with Background Image
def show_home_page():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://wallpapercave.com/wp/wp6871382.jpg');
            background-size: cover;
            background-position: center;
        }
        .home-text {
            color: black;
            font-size: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div class='home-text'>Welcome to our <br> Credit Card Fraud Detection App!</div>", unsafe_allow_html=True)
    if st.button("Go to Login"):
        st.session_state.page = "Login"


# Login Page
def show_login_page():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://wallpapercave.com/wp/wp6871397.jpg');
            background-size: cover;
            background-position: center;
        }
        .home-text {
            color: white;
            font-size: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.write("## Login ")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if validate_login(username, password):
            st.session_state.logged_in = True
            st.session_state.page = "Prediction"
        else:
            st.error("Invalid username or password")


# Prediction Page
def show_prediction_page():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://wallpapercave.com/wp/wp6230306.jpg');
            background-size: cover;
            background-position: center;
        }
        .home-text {
            color: white;
            font-size: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.write("## Credit Card Fraud Detection Model ")
    st.write(" To check the transaction is legitimate or fraudulent")
    input_str = st.text_area(" Enter Required Features (comma-separated)")


    if st.button("Submit"):
        try:
            # Convert input string to a NumPy array of floats
            input_features = np.asarray(input_str.split(','), dtype=np.float64)

            # Reshape the array to fit the model's expected input shape
            features = input_features.reshape(1, -1)

            # Use the loaded model to make a prediction
            prediction = model.predict(features)

            # Interpret the prediction result
            result = "Legitimate Transaction" if prediction[0] == 0 else "Fraudulent Transaction"

            # Store the prediction result in the session state
            st.session_state.prediction = result
            st.session_state.page = "Result"
        except ValueError as e:
            st.error(f"Error processing input: {e}")


# Results Page
def show_results_page():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://wallpapercave.com/wp/wp6871396.jpg');
            background-size: cover;
            background-position: center;
        }
        .home-text {
            color: black;
            font-size: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.write("## PREDICTED RESULT :")
    prediction = st.session_state.get("prediction", "No prediction made.")
    st.write(f"Prediction: {prediction}")


# Main function to control the flow
def main():
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.page == "Home":
        show_home_page()
    elif st.session_state.page == "Login":
        show_login_page()
    elif st.session_state.page == "Prediction":
        if st.session_state.logged_in:
            show_prediction_page()
        else:
            st.error("Please login to access the prediction page.")
    elif st.session_state.page == "Result":
        show_results_page()


if __name__ == "__main__":
    main()
