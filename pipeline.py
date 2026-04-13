from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from preprocessor import FraudPreprocessor

log_reg_pipeline = Pipeline([
    ("preprocess", FraudPreprocessor()),
    ("model", LogisticRegression(max_iter=1000))
])

rf_pipeline = Pipeline([
    ("preprocess", FraudPreprocessor()),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

gb_pipeline = Pipeline([
    ("preprocess", FraudPreprocessor()),
    ("model", GradientBoostingClassifier(n_estimators=200, random_state=42))
])