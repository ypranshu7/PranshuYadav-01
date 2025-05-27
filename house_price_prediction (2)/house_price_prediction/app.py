import numpy as np
from flask import Flask, render_template, request
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model (ensure this path is correct for your model)
model = joblib.load(r'C:\Users\hp pc\house_price_prediction\model\house_price_model_rf.pkl')  # Update path here

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Getting user input from the form
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        stories = int(request.form['stories'])
        
        # Convert categorical inputs to numeric
        mainroad = 1 if request.form['mainroad'] == 'yes' else 0
        guestroom = 1 if request.form['guestroom'] == 'yes' else 0
        basement = 1 if request.form['basement'] == 'yes' else 0
        hotwater = 1 if request.form['hotwater'] == 'yes' else 0
        airconditioning = 1 if request.form['airconditioning'] == 'yes' else 0
        parking = 1 if request.form['parking'] == 'yes' else 0
        prefarea = 1 if request.form['prefarea'] == 'yes' else 0
        
        # Convert 'furnishingstatus' to a numeric value (assuming it has two categories 'furnished' and 'unfurnished')
        furnishingstatus = 1 if request.form['furnishingstatus'] == 'furnished' else 0
        
        # Feature list for prediction (make sure it's in the same order as the training data)
        features = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
                             hotwater, airconditioning, parking, prefarea, furnishingstatus]])

        # Prediction (make sure features is 2D)
        prediction = model.predict(features)  # This will work if the model is loaded correctly
        predicted_price = prediction[0]

        return render_template('index.html', prediction_text=f'Predicted Price: ₹{predicted_price:,.2f}')

if __name__ == "__main__":
    app.run(debug=True)









import numpy as np
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load(r'C:\Users\hp pc\house_price_prediction\model\house_price_model_rf.pkl')  # Ensure correct path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get inputs from the form
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        stories = int(request.form['stories'])

        # Binary encoding for yes/no fields
        mainroad = 1 if request.form['mainroad'].lower() == 'yes' else 0
        guestroom = 1 if request.form['guestroom'].lower() == 'yes' else 0
        basement = 1 if request.form['basement'].lower() == 'yes' else 0
        hotwater = 1 if request.form['hotwater'].lower() == 'yes' else 0
        airconditioning = 1 if request.form['airconditioning'].lower() == 'yes' else 0
        parking = 1 if request.form['parking'].lower() == 'yes' else 0
        prefarea = 1 if request.form['prefarea'].lower() == 'yes' else 0

        # Encoding furnishing status
        furnishingstatus = request.form['furnishingstatus'].lower()
        if furnishingstatus == 'furnished':
            furnishing_val = 2
        elif furnishingstatus == 'semi-furnished':
            furnishing_val = 1
        else:
            furnishing_val = 0

        # Prepare input for model
        features = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
                              hotwater, airconditioning, parking, prefarea, furnishing_val]])

        # Predict
        predicted_price = model.predict(features)[0]

        # Formatted prediction text
        prediction_text = f"The predicted price for the house based on your selected features is ₹{predicted_price:,.2f}."

        return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
