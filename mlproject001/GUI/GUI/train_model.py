import os
os.makedirs("models", exist_ok=True)

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("C:\\Users\\nk568\\OneDrive\\Desktop\\mlproject001\\Letterbox Movie Classification\\Letterbox Movie Classification Dataset.csv")

# Target creation
def categorize_rating(score):
    if score >= 3.8: return 'High'
    elif score >= 3.0: return 'Medium'
    return 'Low'

df['Rating_Class'] = df['Average_rating'].apply(categorize_rating)

# Encode text columns
le_dict = {}
text_features = ['Film_title', 'Director', 'Genres', 'Original_language', 'Description', 'Studios']

for col in text_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le  # Save encoders if needed later

# Features
features = [
    'Unnamed: 0', 'Film_title', 'Director', 'Genres', 'Runtime', 
    'Original_language', 'Description', 'Studios', 'Watches', 
    'List_appearances', 'Likes', 'Fans', 'Lowest★', 
    'Medium★★★', 'Highest★★★★★', 'Total_ratings'
]

X = df[features]
y = df['Rating_Class']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 SCALE DATA (Fix for convergence)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

accuracies = {}

for name, model in models.items():

    # Use scaled data only for LR & SVM
    if name in ["Logistic Regression", "SVM"]:
        model.fit(X_train_scaled, y_train)
        acc = model.score(X_test_scaled, y_test)
    else:
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

    accuracies[name] = acc

    # Save model
    joblib.dump(model, f'models/{name.lower().replace(" ", "_")}_model.pkl')

# Save scaler + accuracies
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(accuracies, 'models/accuracies.pkl')

print("Training completed successfully!")
print("Accuracies:", accuracies)