import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Feature and target from your CSV
model_df = df[['Hours_Played_Weekly', 'Game_Purchase_Amount']].dropna()

X = model_df[['Hours_Played_Weekly']].values
y = model_df['Game_Purchase_Amount'].values

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict purchase amount
y_pred = model.predict(X)

# Display the regression equation
slope = model.coef_[0]
intercept = model.intercept_

print("Linear Regression Equation:")
print(f"Game Purchase Amount = {slope:.3f} × Hours Played Weekly + {intercept:.3f}")

# Model evaluation
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\nMean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Predict purchase amount for a new hours-played value
new_input = np.array([[45]])
predicted_output = model.predict(new_input)

print(f"\nPredicted purchase amount for 45 hours played/week: "
      f"₹{predicted_output[0]:.2f}")

# Plot experimental data and regression line
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", label="Regression Line")

plt.xlabel("Hours Played Weekly")
plt.ylabel("Game Purchase Amount (₹)")
plt.title("Linear Regression: Hours Played vs Purchase Amount")
plt.legend()
plt.grid(True)
plt.show()
