# ── 0. Upload ─────────────────────────────────────────────────────────────
import os
if os.path.exists('gaming_players_data.csv'):
    os.remove('gaming_players_data.csv')

from google.colab import files
uploaded = files.upload()
uploaded_filename = list(uploaded.keys())[0]
print("Uploaded file(s):", uploaded_filename)

# -----------------------------------------
# 1. Load dataset
# -----------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

data = pd.read_csv(uploaded_filename)
print("Dataset loaded successfully!")
print(data.head())

# -----------------------------------------
# 2. Create a binary target + select features
# -----------------------------------------
# No ready-made binary column, so we create one:
# 1 = spent above median purchase amount, 0 = below/equal
data = data.dropna(subset=['Game_Purchase_Amount', 'Hours_Played_Weekly', 'Age'])
median_spend = data['Game_Purchase_Amount'].median()
data['High_Spender'] = (data['Game_Purchase_Amount'] > median_spend).astype(int)

X = data[['Hours_Played_Weekly', 'Age']]
y = data['High_Spender']

# -----------------------------------------
# 3. Split into training and testing data
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=15, stratify=y
)

# -----------------------------------------
# 4. Try different tree depths
# -----------------------------------------
depths = [3, 4, 5]
results = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=15)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append([depth, accuracy, precision, recall, f1])

# -----------------------------------------
# 5. Comparative analysis
# -----------------------------------------
results_df = pd.DataFrame(
    results,
    columns=["Max Depth", "Accuracy", "Precision", "Recall", "F1 Score"]
)

print("\nDecision Tree Comparison:")
print(results_df.to_string(index=False))

# -----------------------------------------
# 6. Visualize the best tree (pick the depth you like from the table above)
# -----------------------------------------
best_depth = 3  # change this after looking at results_df
best_model = DecisionTreeClassifier(max_depth=best_depth, random_state=15)
best_model.fit(X_train, y_train)

plt.figure(figsize=(16, 8))
plot_tree(best_model, feature_names=['Hours_Played_Weekly', 'Age'],
          class_names=['Low Spender', 'High Spender'], filled=True, rounded=True, fontsize=10)
plt.title(f'Decision Tree (max_depth={best_depth}) — High Spender Classification')
plt.show()
