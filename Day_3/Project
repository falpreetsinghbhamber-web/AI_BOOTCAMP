import os
if os.path.exists('gaming_players_data.csv'):
    os.remove('gaming_players_data.csv')

from google.colab import files
uploaded = files.upload()

# Use whatever filename was actually uploaded, no matter what it's called
uploaded_filename = list(uploaded.keys())[0]
print("Uploaded file(s):", uploaded_filename)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score

df = pd.read_csv(uploaded_filename)
df.head()

# ── 2. Clean & Explore ───────────────────────────────────────────────────
df['Signup_Date'] = pd.to_datetime(df['Signup_Date'], errors='coerce', dayfirst=True)
df['Last_Login_Date'] = pd.to_datetime(df['Last_Login_Date'], errors='coerce', dayfirst=True)

print(df.shape)
print(df.info())
print(df.head())
print(df.isna().sum())
print(df[['Age', 'Hours_Played_Weekly', 'Friend_Count', 'Game_Purchase_Amount']].describe())

dupes = df.drop(columns=['Player_ID']).duplicated(keep=False)
print(f"Duplicate rows (excluding ID): {dupes.sum()}")
df[dupes]

# ── 3. Visualize: Feature vs Target by Category ─────────────────────────
feature = 'Hours_Played_Weekly'
target = 'Game_Purchase_Amount'
category = 'Platform'

plot_df = df[[feature, target, category]].dropna()
categories = plot_df[category].unique()
colors = plt.cm.Set1.colors

plt.figure(figsize=(8, 5))
for i, cat_value in enumerate(categories):
    subset = plot_df[plot_df[category] == cat_value]
    plt.scatter(subset[feature], subset[target],
                color=colors[i % len(colors)], label=cat_value, alpha=0.7)

plt.title(f'{feature} vs {target} by {category}')
plt.xlabel(feature)
plt.ylabel(target)
plt.legend(title=category)
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()

# ── 4. Prepare Features/Target for Modeling (shared by both models) ─────
features = ['Hours_Played_Weekly', 'Age']
target = 'Game_Purchase_Amount'

model_df = df[features + [target]].dropna()
X = model_df[features]
y = model_df[target]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 5. Linear Regression ─────────────────────────────────────────────────
model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)
print("predictions")
print(predictions)
print("x_test")
print(x_test)

accuracy = r2_score(y_test, predictions)
print(f"Linear Regression Accuracy (R² score): {accuracy * 100:.2f}%")

# ── 6. Decision Tree ──────────────────────────────────────────────────────
tree_model = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_model.fit(x_train, y_train)

tree_predictions = tree_model.predict(x_test)
tree_accuracy = r2_score(y_test, tree_predictions)
print(f"Decision Tree Accuracy (R² score): {tree_accuracy * 100:.2f}%")

plt.figure(figsize=(16, 8))
plot_tree(tree_model, feature_names=features, filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree — What the Model Learned')
plt.show()
