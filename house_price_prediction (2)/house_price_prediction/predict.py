import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('model/house_price_model_rf.pkl')

# Initialize LabelEncoder for categorical features
le = LabelEncoder()

# Categorical columns that need encoding
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                    'airconditioning', 'prefarea', 'furnishingstatus']

# Function to get user input
# Function to get user input
def get_user_input():
    print("Enter the house details:")
    
    area = float(input("Area (sqft): "))
    bedrooms = int(input("Bedrooms: "))
    bathrooms = int(input("Bathrooms: "))
    stories = int(input("Stories: "))
    mainroad = input("Mainroad (yes/no): ")
    guestroom = input("Guestroom (yes/no): ")
    basement = input("Basement (yes/no): ")
    hotwaterheating = input("Hotwater Heating (yes/no): ")
    airconditioning = input("Airconditioning (yes/no): ")
    parking = input("Parking (yes/no): ")
    prefarea = input("Prefarea (yes/no): ")
    furnishingstatus = input("Furnishing status (furnished/semi-furnished/unfurnished): ")
    
    # Convert yes/no input to 1/0 for parking
    if parking.lower() == 'yes':
        parking = 1
    else:
        parking = 0
    
    # Create a DataFrame for the input data
    user_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [mainroad],
        'guestroom': [guestroom],
        'basement': [basement],
        'hotwaterheating': [hotwaterheating],
        'airconditioning': [airconditioning],
        'parking': [parking],
        'prefarea': [prefarea],
        'furnishingstatus': [furnishingstatus]
    })
    
    # Encode categorical columns the same way as in the training data
    for col in categorical_cols:
        user_data[col] = le.fit_transform(user_data[col])

    return user_data


# Get the input from the user
input_data = get_user_input()

# Predict the price using the trained model
predicted_price = model.predict(input_data)

# Display the predicted price
print(f"Predicted price for the entered house: ₹{predicted_price[0]:.2f}")
