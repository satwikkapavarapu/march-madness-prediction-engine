import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')

# Load datasets
tourney = pd.read_csv(os.path.join(DATA_DIR, 'ncaa_tournament_history.csv'))
kenpom = pd.read_csv(os.path.join(DATA_DIR, 'kenpom.csv'))
odds = pd.read_csv(os.path.join(DATA_DIR, 'odds.csv'))

print('Tournament shape:', tourney.shape)
print('KenPom shape:', kenpom.shape)
print('Odds shape:', odds.shape)

print('Tournament columns:', tourney.columns.tolist())
print('KenPom columns:', kenpom.columns.tolist())
print('Odds columns:', odds.columns.tolist())
