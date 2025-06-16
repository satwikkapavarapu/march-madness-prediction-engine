import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
FEATURES_CSV = os.path.join(DATA_DIR, 'model_features.csv')

# Load features
df = pd.read_csv(FEATURES_CSV)
print(df.head())
print(df.info())
print(df.describe(include='all'))
