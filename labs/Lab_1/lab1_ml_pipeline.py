# =================================================
# LAB1 MLOPS - PIPELINE COMPLET
# Étudiant: Ymed - Deadline: 17h00
# =================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

print("🚀 LAB1 MLOPS - DÉMARRAGE RÉUSSI!")
print("⏰ Il est 14h31 - Deadline: 17h00")
print("="*60)

# 1. DATA LOADING
def get_data():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    df = pd.concat([data['data'], data['target']], axis=1)
    return df

print("📊 Chargement des données...")
data = get_data()
X = data.drop("target", axis=1)
y = data["target"]
print(f"✅ Dataset: {data.shape} - Features: {X.shape[1]}")

# 2. FEATURE ENGINEERING
def add_combined_feature(X):
    X = X.copy()
    X['Combined_radius_texture'] = X['mean radius'] * X['mean texture']
    return X

def add_advanced_features(X):
    X = X.copy()
    X['Combined_radius_texture'] = X['mean radius'] * X['mean texture']
    X['radius_texture_ratio'] = X['mean radius'] / (X['mean texture'] + 0.001)
    X['area_perimeter_ratio'] = X['mean area'] / (X['mean perimeter'] + 0.001)
    return X

print("✅ Feature engineering défini!")

# 3. MODEL SETUP & TRAINING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"✅ Split - Train: {X_train.shape}, Test: {X_test.shape}")

models_config = [
    {
        'name': 'Logistic_Regression',
        'pipeline': Pipeline([
            ('features', FunctionTransformer(add_combined_feature)),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'params': {'classifier__C': [0.1, 1.0, 10]}
    },
    {
        'name': 'Random_Forest',
        'pipeline': Pipeline([
            ('features', FunctionTransformer(add_advanced_features)),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        'params': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20]
        }
    },
    {
        'name': 'SVM',
        'pipeline': Pipeline([
            ('features', FunctionTransformer(add_advanced_features)),
            ('scaler', StandardScaler()),
            ('classifier', SVC(probability=True, random_state=42))
        ]),
        'params': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf']
        }
    }
]

print("\n🎯 ENTRAÎNEMENT DES 3 MODÈLES...")
print("⚠️  Durée: 10-15 minutes - Patience !")

results = {}
best_models = {}

for i, config in enumerate(models_config, 1):
    print(f"\n[{i}/3] 🚀 {config['name']}...")
    
    grid_search = GridSearchCV(
        config['pipeline'], 
        config['params'], 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_models[config['name']] = grid_search.best_estimator_
    test_score = grid_search.best_estimator_.score(X_test, y_test)
    
    results[config['name']] = {
        'cv_score': grid_search.best_score_,
        'test_score': test_score,
        'best_params': grid_search.best_params_
    }
    
    print(f"   ✅ CV: {grid_search.best_score_:.4f}, Test: {test_score:.4f}")

# 4. BEST MODEL SELECTION
best_model_name = max(results.keys(), key=lambda x: results[x]['test_score'])
best_model = best_models[best_model_name]

print(f"\n🏆 COMPARAISON FINALE:")
print("="*50)
for name in results:
    score = results[name]['test_score']
    marker = "🥇" if name == best_model_name else "   "
    print(f"{marker} {name}: {score:.4f}")

print(f"\n🎉 MEILLEUR MODÈLE: {best_model_name}")

# 5. EVALUATION
y_pred = best_model.predict(X_test)
print(f"\n📊 PERFORMANCE FINALE:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. SAUVEGARDE
dump(best_model, 'best_cancer_model_pipeline.joblib')
print(f"\n💾 MODÈLE SAUVEGARDÉ !")

# Test chargement
loaded_model = load('best_cancer_model_pipeline.joblib')
test_pred = loaded_model.predict(X_test[:1])
print(f"✅ Test OK - Prédiction: {test_pred[0]}")

print("\n" + "="*60)
print("🎉 LAB1 PIPELINE TERMINÉ AVEC SUCCÈS!")
print("🚀 Prêt pour l'API Flask!")
print("="*60)
