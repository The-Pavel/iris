from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def create_pipeline():
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())])
    return pipeline
