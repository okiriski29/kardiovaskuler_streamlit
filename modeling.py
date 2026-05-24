#Import Library
import numpy as np
import pandas as pd
import pickle
#Import Library untuk Klasifikasi
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# ----- Pengerjaan Model -----
data =pd.read_csv("Kardio.csv")
print(data.head())
#cleaning the data by dropping unneccessary column and dividing the data as features(x3) & target(y3)
X = data.drop(columns=['kardio'])
y = data['kardio']
# ----- 3. Split Data ----- #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# ----- 4. GridSearchCV ----- #
rf = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf= 1, min_samples_split=2, random_state=42)

best_model = rf.fit(X_train, y_train)
# Evaluasi model terbaik pada data testing
y_test_pred_best_rf = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_test_pred_best_rf)

print(f"Accuracy pada data testing dengan model Random Forest terbaik: {final_accuracy:.4f}")
filename = "modelkardio.pkl"
with open(filename, 'wb') as file:
    pickle.dump(best_model, file)