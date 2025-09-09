# =================================================
# LAB1 MLOPS - PIPELINE COMPLET AVEC GRAPHIQUES
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

print("🚀 LAB1 MLOPS - DÉMARRAGE AVEC GRAPHIQUES !")
print("⏰ Il est 14h52 - Deadline: 17h00")
print("="*60)

# =================================================
# 1. DATA COLLECTION
# =================================================

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

# =================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =================================================

print("\n📈 Analyse exploratoire avec graphiques...")

plt.figure(figsize=(15, 5))

# Distribution du target
plt.subplot(1, 3, 1)
sns.countplot(data=data, x='target')
plt.title('Distribution Target\n(0=Malignant, 1=Benign)')

# Feature importance
mi_scores = mutual_info_classif(X, y, random_state=42)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': mi_scores
}).sort_values('Importance', ascending=False)

plt.subplot(1, 3, 2)
sns.barplot(data=feature_importance.head(10), y='Feature', x='Importance', palette='viridis')
plt.title('Top 10 Features Importance')

# Box plot
plt.subplot(1, 3, 3)
data[['mean radius', 'mean texture', 'target']].boxplot(by='target')
plt.suptitle('')
plt.title('Mean Radius par Target')

plt.tight_layout()
plt.show()  # ← GRAPHIQUES EDA AFFICHÉS !

print("✅ EDA terminée!")

# =================================================
# 3. FEATURE ENGINEERING
# =================================================

print("\n🔧 Définition Feature Engineering...")

def add_combined_feature(X):
    """Feature engineering pour Logistic Regression"""
    X = X.copy()
    X['Combined_radius_texture'] = X['mean radius'] * X['mean texture']
    return X

def add_advanced_features(X):
    """Feature engineering avancée pour Random Forest et SVM"""
    X = X.copy()
    
    # Feature de base (cohérence)
    X['Combined_radius_texture'] = X['mean radius'] * X['mean texture']
    
    # Ratios significatifs
    X['radius_texture_ratio'] = X['mean radius'] / (X['mean texture'] + 0.001)
    X['area_perimeter_ratio'] = X['mean area'] / (X['mean perimeter'] + 0.001)
    X['compactness_concavity'] = X['mean compactness'] * X['mean concavity']
    
    # Features polynomiales
    X['radius_squared'] = X['mean radius'] ** 2
    X['texture_smoothness'] = X['mean texture'] * X['mean smoothness']
    
    return X

print("✅ Fonctions de feature engineering définies!")

# =================================================
# 4. DATA PREPROCESSING & MODEL SETUP
# =================================================

print("\n📦 Préparation des données et modèles...")

# Split stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"✅ Split des données:")
print(f"   - Train: {X_train.shape}")
print(f"   - Test: {X_test.shape}")

# Configuration des 3 modèles requis
models_config = [
    {
        'name': 'Logistic_Regression',
        'pipeline': Pipeline([
            ('features', FunctionTransformer(add_combined_feature)),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'params': {
            'classifier__C': [0.1, 1.0, 10]
        }
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

print("✅ Configuration des 3 modèles terminée!")

# =================================================
# 5. MODEL TRAINING & HYPERPARAMETER TUNING
# =================================================

print("\n🎯 ENTRAÎNEMENT DES 3 MODÈLES avec GridSearchCV")
print("⚠️  Durée estimée: 10-15 minutes - Patience !")
print("="*60)

results = {}
best_models = {}

for i, config in enumerate(models_config, 1):
    print(f"\n[{i}/3] 🚀 Entraînement: {config['name']}")
    
    # GridSearchCV
    grid_search = GridSearchCV(
        config['pipeline'], 
        config['params'], 
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1,  # Parallélisation
        verbose=0
    )
    
    # Entraînement
    grid_search.fit(X_train, y_train)
    
    # Évaluation sur test set
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    
    # Stockage résultats
    best_models[config['name']] = best_model
    results[config['name']] = {
        'cv_score': grid_search.best_score_,
        'test_score': test_score,
        'best_params': grid_search.best_params_
    }
    
    print(f"   ✅ CV Score: {grid_search.best_score_:.4f}")
    print(f"   ✅ Test Score: {test_score:.4f}")
    print(f"   📋 Meilleurs params: {grid_search.best_params_}")

print("\n🎉 ENTRAÎNEMENT TERMINÉ!")

# =================================================
# 6. MODEL EVALUATION & COMPARISON
# =================================================

print("\n📊 ÉVALUATION ET COMPARAISON DES MODÈLES")
print("="*60)

# Sélection du meilleur modèle
best_model_name = max(results.keys(), key=lambda x: results[x]['test_score'])
best_model = best_models[best_model_name]

print(f"🏆 MEILLEUR MODÈLE: {best_model_name}")
for name in results:
    score = results[name]['test_score']
    marker = "🥇" if name == best_model_name else "   "
    print(f"{marker} {name}: {score:.4f}")

# Évaluation détaillée du meilleur modèle
y_pred = best_model.predict(X_test)
print(f"\n📊 PERFORMANCE DÉTAILLÉE - {best_model_name}:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# =================================================
# 7. VISUALISATION DES RÉSULTATS
# =================================================

print("\n📈 Génération des graphiques de résultats...")

plt.figure(figsize=(15, 6))

# Comparaison des scores
plt.subplot(1, 2, 1)
model_names = list(results.keys())
cv_scores = [results[name]['cv_score'] for name in model_names]
test_scores = [results[name]['test_score'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, cv_scores, width, label='CV Score', alpha=0.8, color='skyblue')
plt.bar(x + width/2, test_scores, width, label='Test Score', alpha=0.8, color='lightcoral')

plt.xlabel('Modèles')
plt.ylabel('Accuracy')
plt.title('Comparaison des Performances')
plt.xticks(x, model_names, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Confusion Matrix du meilleur modèle
plt.subplot(1, 2, 2)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Malignant', 'Benign'], 
            yticklabels=['Malignant', 'Benign'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Prédictions')
plt.ylabel('Vraies valeurs')

plt.tight_layout()
plt.show()  # ← GRAPHIQUES DE RÉSULTATS AFFICHÉS !

# =================================================
# 8. SAUVEGARDE ET TESTS FINAUX
# =================================================

print("\n💾 SAUVEGARDE DU MEILLEUR MODÈLE")
model_filename = 'best_cancer_model_pipeline.joblib'
dump(best_model, model_filename)
print(f"✅ Modèle sauvegardé: {model_filename}")

# Test de chargement
try:
    loaded_model = load(model_filename)
    test_prediction = loaded_model.predict(X_test[:1])
    print(f"✅ Test de chargement réussi - Prédiction: {test_prediction[0]}")
    
    # Test sur plusieurs échantillons
    test_samples = X_test.head(3)
    predictions = loaded_model.predict(test_samples)
    print(f"\n🧪 Tests sur 3 échantillons:")
    for i, pred in enumerate(predictions):
        label = "Benign" if pred == 1 else "Malignant"
        print(f"   Échantillon {i+1}: {label}")
        
except Exception as e:
    print(f"❌ Erreur lors du test: {e}")

# =================================================
# 9. RÉSUMÉ FINAL
# =================================================

print("\n" + "="*60)
print("🎉 LAB1 MLOPS - TERMINÉ AVEC SUCCÈS !")
print("="*60)
print(f"✅ Dataset traité: {data.shape[0]} échantillons, {X.shape[1]} features")
print(f"✅ 3 modèles entraînés: Logistic Regression, Random Forest, SVM")  
print(f"✅ Hyperparameter tuning avec GridSearchCV (5-fold CV)")
print(f"✅ Meilleur modèle: {best_model_name} ({results[best_model_name]['test_score']:.4f})")
print(f"✅ Modèle sauvegardé pour déploiement")
print(f"✅ Graphiques EDA et résultats générés")
print("="*60)
print("🚀 PRÊT POUR LE DÉPLOIEMENT AVEC FLASK API!")
print("📊 GRAPHIQUES AFFICHÉS COMME TES POTES !")
print("="*60)
