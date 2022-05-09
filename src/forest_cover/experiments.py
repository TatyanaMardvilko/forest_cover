from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

models = []

models.append("Logistic Regression:", LogisticRegression())
models.append("Random Forest:", RandomForestClassifier())