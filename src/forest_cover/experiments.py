import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

models = []

models.append("Logistic Regression:", LogisticRegression())
models.append("Random Forest:", RandomForestClassifier(n_estimators=10))

param_grid_log_res = {'C': np.logspace(-4, 4, 50), 'penalty': ['l1', 'l2']}

models.append("Logistic Regression with GridSearch:",
              GridSearchCV(LogisticRegression(random_state=0), param_grid_log_res, cv=5, verbose=0, n_jobs=-1))

n_estimators = [7, 10, 15]
max_features = ['sqrt']
max_depth = [2, 3, 7, 11, 15]
min_samples_split = [2, 3, 4, 22, 23, 24]
min_samples_leaf = [2, 3, 4, 5, 6, 7]
bootstrap = [False]
param_grid_forest = {'n_estimators': n_estimators,
                     'max_features': max_features,
                     'max_depth': max_depth,
                     'min_samples_split': min_samples_split,
                     'min_samples_leaf': min_samples_leaf,
                     'bootstrap': bootstrap}

models.append("Logistic Regression with GridSearch:",
              GridSearchCV(RandomForestClassifier(), param_grid_forest, cv=5, verbose=0, n_jobs=-1))
