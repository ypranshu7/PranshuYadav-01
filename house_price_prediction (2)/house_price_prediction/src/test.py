import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Step 1: Load CSV
df = pd.read_csv("C:/Users/hp pc/house_price_prediction/data/house_prices.csv")

# Step 2: Handle missing values
# Filling missing numerical values with the column mean
df.fillna(df.select_dtypes(include=['float64', 'int64']).mean(), inplace=True)
# Filling missing categorical values with the mode (most frequent value)
df.fillna(df.select_dtypes(include=['object']).mode().iloc[0], inplace=True)

# Step 3: Encoding Categorical Variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 4: Train-Test Split
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 5: Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 6: Accuracy
accuracy = r2_score(y_test, y_pred)
print(f"Model Accuracy (R²): {accuracy * 100:.2f}%")  # Displaying R² as the accuracy

# Step 7: Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")

# Step 8: Visualizations

# Area vs Price (Scatter plot)
plt.figure(figsize=(8,5))
plt.scatter(df['area'], df['price'], color='blue', alpha=0.6)
plt.xlabel('Area (sqft)')
plt.ylabel('Price (INR)')
plt.title('Area vs Price')
plt.grid(True)
plt.show()

# Furnishing Status Count (Bar plot)
plt.figure(figsize=(6,4))
sns.countplot(x='furnishingstatus', data=df)
plt.title('Furnishing Status Count')
plt.show()

# Price Distribution (Histogram)
plt.figure(figsize=(8,4))
sns.histplot(df['price'], bins=20, kde=True, color='green')
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Count')
plt.show()

# Feature Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()

# Price vs Number of Bedrooms (Box plot)
plt.figure(figsize=(6,4))
sns.boxplot(x='bedrooms', y='price', data=df)
plt.title('Price vs Number of Bedrooms')
plt.show()

# Actual vs Predicted Price (Scatter plot with line)
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Price")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Line of perfect prediction
plt.show()
