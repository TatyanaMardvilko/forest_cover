import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from src.forest_cover.data import get_dataset
from sklearn.model_selection import RandomizedSearchCV

features_train, features_val, target_train, target_val = get_dataset(
    'd:\\maschineLearning\RS\\ml_hometask9\data\\train.csv',
    42,
    0.2,
)

scaler = StandardScaler()
features_tr_sc = scaler.fit_transform(features_train)

models = []

models.append(("Logistic Regression:", LogisticRegression(max_iter=1000)))
models.append(("Random Forest:", RandomForestClassifier(n_estimators=10)))

param_grid_log_res = {'C': np.logspace(-4, 4, 10)}

models.append(("Logistic Regression with l1:",
              LogisticRegression(random_state=0, max_iter=1000, penalty="l1", C=0.1, solver='liblinear')))

models.append(("Logistic Regression with elasticnet:",
              LogisticRegression(random_state=1, max_iter=1000, solver='saga', penalty='elasticnet', l1_ratio=0.5, C=0.01)))

n_estimators = [7, 10, 15]
max_features = ['auto', 'sqrt']
max_depth = [2, 7, 11]
min_samples_split = [2, 4, 22]
min_samples_leaf = [1, 2, 3, 4, 5, 6, 7]
bootstrap = [False]
param_grid_forest = {'n_estimators': n_estimators,
                     'max_features': max_features,
                     'max_depth': max_depth,
                     'min_samples_split': min_samples_split,
                     'min_samples_leaf': min_samples_leaf,
                     'bootstrap': bootstrap}

models.append(("Random Forest with GridSearch:",
              GridSearchCV(RandomForestClassifier(), param_grid_forest, cv=5, verbose=0, n_jobs=-1)))

models.append(("Random Forest with RandomSearch:",
              RandomizedSearchCV(RandomForestClassifier(), param_grid_forest, cv=5, verbose=0, n_jobs=-1)))

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=0, shuffle=True)
    cv_result = cross_val_score(model, features_tr_sc, target_train, cv=kfold, scoring='accuracy')
    names.append(name)
    print(name)
    results.append(cv_result)
for i in range(len(names)):
    print(names[i], results[i].mean())
