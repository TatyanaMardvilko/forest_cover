from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from src.forest_cover.data import get_dataset
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

features_train, features_val, target_train, target_val = get_dataset(
    'd:\\maschineLearning\RS\\ml_hometask9\data\\train.csv',
    42,
    0.2,
)

cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)
model = RandomForestClassifier(random_state=1)

n_estimators = [7, 10, 15]
max_depth = [2, 7, 11]
min_samples_split = [2, 4, 22]
min_samples_leaf = [1, 3, 5, 7]
bootstrap = [False]
param_grid_forest = {'n_estimators': n_estimators,
                     'max_depth': max_depth,
                     'min_samples_split': min_samples_split,
                     'min_samples_leaf': min_samples_leaf,
                     'bootstrap': bootstrap}

search = GridSearchCV(model, param_grid_forest, scoring='accuracy', n_jobs=1, cv=cv_inner, refit=True)
cv_outer = KFold(n_splits=6, shuffle=True, random_state=1)
#accuracies = cross_val_score(search, features_train, target_train, scoring='accuracy', cv=cv_outer, n_jobs=-1)
best_model = search.fit(features_train, target_train).best_estimator_
print(best_model)
f1_scores = f1_score(target_val, best_model.predict(features_val), average=None)
print(f1_scores)
roc_auc_scores = roc_auc_score(target_val, best_model.predict_proba(features_val), multi_class='ovr')
print(roc_auc_scores)
#f1_scores = cross_val_score(search, features_train, target_train, scoring='f1', cv=cv_outer, n_jobs=-1)
#roc_auc_scores = cross_val_score(search, features_train, target_train, scoring='roc_auc', cv=cv_outer, n_jobs=-1)

#print(accuracies.mean, f1_scores.mean, roc_auc_scores.mean)