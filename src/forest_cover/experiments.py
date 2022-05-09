import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

models = []

models.append("Logistic Regression:", LogisticRegression())
models.append("Random Forest:", RandomForestClassifier())

param_grid = {'C': np.logspace(-4, 4, 50), 'penalty': ['l1', 'l2']}

models.append("Logistic Regression with GridSearch:",
              GridSearchCV(LogisticRegression(random_state=0), param_grid, cv =5, verbose = 0, n_jobs = -1))