import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from src.forest_cover.data import get_dataset
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
import mlflow.sklearn

features_train, features_val, target_train, target_val = get_dataset(
    "d:\\maschineLearning\RS\\ml_hometask9\data\\train.csv",
    42,
    0.2,
)

scaler = StandardScaler()
features_tr_sc = scaler.fit_transform(features_train)

pca = PCA(n_components=10)
X_train_scaled_pca = pca.fit_transform(features_tr_sc)

models = []

models.append(("Logistic Regression:", LogisticRegression(max_iter=1000)))
models.append(("Random Forest:", RandomForestClassifier(n_estimators=10)))

param_grid_log_res = {"C": np.logspace(-4, 4, 10)}

models.append(
    (
        "Logistic Regression with l1:",
        LogisticRegression(
            random_state=0, max_iter=1000, penalty="l1", C=0.1, solver="liblinear"
        ),
    )
)

models.append(("Logistic Regression with elasticnet:",
               LogisticRegression(random_state=1, max_iter=1000, solver='saga',
                                  penalty='elasticnet', l1_ratio=0.5, C=0.01)))

n_estimators = [7, 10, 15]
max_features = ["sqrt"]
max_depth = [2, 7, 11]
min_samples_split = [2, 4, 22]
min_samples_leaf = [1, 2, 3, 4, 5, 6, 7]
bootstrap = [False]
param_grid_forest = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "bootstrap": bootstrap,
}

models.append(
    (
        "Random Forest with GridSearch:",
        GridSearchCV(
            RandomForestClassifier(), param_grid_forest, cv=5, verbose=0, n_jobs=-1
        ),
    )
)

models.append(
    (
        "Random Forest with RandomSearch:",
        RandomizedSearchCV(
            RandomForestClassifier(), param_grid_forest, cv=5, verbose=0, n_jobs=-1
        ),
    )
)

exp = mlflow.create_experiment("forest")
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=0, shuffle=True)
    cv_result = cross_val_score(
        model, features_tr_sc, target_train, cv=kfold, scoring="accuracy"
    )
    names.append(name)
    print(name)
    results.append(cv_result)
for name, model in models:
    cv_result = cross_val_score(
        model, X_train_scaled_pca, target_train, cv=kfold, scoring="accuracy"
    )
    names.append(name.replace(":", " PCA:"))
    print(name.replace(":", " PCA:"))
    results.append(cv_result)
for i in range(len(names)):
    print(names[i], results[i].mean())
    if i < len(models) - 1:
        model_exp = models[i][1]
    else:
        model_exp = models[i - len(models)][1]
    with mlflow.start_run(experiment_id="1", run_name=names[i]):
        mlflow.log_param("model_name", names[i])
        mlflow.log_metric("accuracy", results[i].mean())
        mlflow.log_param("scaler", StandardScaler())
        if "Logistic Regression" in names[i]:
            print(names[i], model_exp)
            mlflow.log_param("max_iter", model_exp.max_iter)
            mlflow.log_param("C", model_exp.C)
            mlflow.log_param("penalty", model_exp.penalty)
        elif "GridSearch" in names[i]:
            mlflow.log_param(
                "min_samples_leaf", model_exp.param_grid["min_samples_leaf"]
            )
            mlflow.log_param("max_depth", model_exp.param_grid["max_depth"])
            mlflow.log_param("max_features", model_exp.param_grid["max_features"])
            mlflow.log_param("n_estimators", model_exp.param_grid["n_estimators"])
        elif "RandomSearch" in names[i]:
            mlflow.log_param(
                "min_samples_leaf", model_exp.param_distributions["min_samples_leaf"]
            )
            mlflow.log_param("max_depth", model_exp.param_distributions["max_depth"])
            mlflow.log_param(
                "max_features", model_exp.param_distributions["max_features"]
            )
            mlflow.log_param(
                "n_estimators", model_exp.param_distributions["n_estimators"]
            )
        else:
            print(names[i])
            mlflow.log_param("criterion", model_exp.criterion)
            mlflow.log_param("max_depth", model_exp.max_depth)
            mlflow.log_param("max_features", model_exp.max_features)
            mlflow.log_param("n_estimators", model_exp.n_estimators)
