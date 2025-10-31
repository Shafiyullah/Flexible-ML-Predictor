import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # X is expected to be a DataFrame of datetime columns
        df = pd.DataFrame(X)
        output_df = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            # Ensure it's datetime
            dt_col = pd.to_datetime(df[col], errors='coerce')
            
            output_df[f'{col}_month'] = dt_col.dt.month
            output_df[f'{col}_dayofweek'] = dt_col.dt.dayofweek
            output_df[f'{col}_dayofyear'] = dt_col.dt.dayofyear
            output_df[f'{col}_hour'] = dt_col.dt.hour
            output_df[f'{col}_minute'] = dt_col.dt.minute
        
        output_df = output_df.fillna(0)
        return output_df

def create_preprocessor(numeric_features: list, categorical_features: list, datetime_features: list) -> ColumnTransformer:    
    # Numeric Pipeline
    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    )

    # Categorical Pipeline
    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
        ]
    )
    
    # Datetime Pipeline
    datetime_transformer = Pipeline(
        steps=[
            ('dt_extractor', DatetimeFeatureExtractor()),
            ('scaler', StandardScaler())
        ]
    )

    # Combine Pipelines in a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('dt', datetime_transformer, datetime_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor