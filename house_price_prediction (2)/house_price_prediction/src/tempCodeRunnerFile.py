import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 1: Load CSV
df = pd.read_csv("C:/Users/hp pc/house_price_prediction/data/house_prices.csv")

# Step 2: Encoding Categorical Variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 3: Train-Test Split
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 4: Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 5: Accuracy
accuracy = r2_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 6: Visualizations
plt.figure(figsize=(8,5))
plt.scatter(df['area'], df['price'], color='blue')
plt.xlabel('Area (sqft)')
plt.ylabel('Price (INR)')
plt.title('Area vs Price')
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='furnishingstatus', data=df)
plt.title('Furnishing Status Count')
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(df['price'], bins=10, kde=True, color='green')
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='bedrooms', y='price', data=df)
plt.title('Price vs Number of Bedrooms')
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Price")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
