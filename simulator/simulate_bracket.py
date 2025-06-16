# Monte Carlo simulation engine for NCAA bracket prediction
import pandas as pd
import numpy as np
import joblib
import os
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
CALIBRATED_MODEL_OUT = os.path.join(DATA_DIR, 'xgb_model_calibrated.pkl')
FEATURES_CSV = os.path.join(DATA_DIR, 'model_features.csv')

# ESPN-style scoring rules
ROUND_POINTS = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160, 6: 320}

# Helper: get all teams for a given year

def get_teams(df, year):
    teams = set(df[df['year'] == year]['favorite']).union(set(df[df['year'] == year]['underdog']))
    return sorted(list(teams))

# Helper: get all matchups for a round
def get_matchups(df, year, rnd):
    return df[(df['year'] == year) & (df['round_num'] == rnd)][['favorite', 'underdog']].values.tolist()

# Simulate a single bracket
def simulate_bracket(df, model, year, n_rounds=6):
    bracket = []
    winners = {}
    for rnd in range(1, n_rounds+1):
        matchups = get_matchups(df, year, rnd)
        rnd_winners = []
        for fav, und in matchups:
            row = df[(df['year'] == year) & (df['round_num'] == rnd) & (df['favorite'] == fav) & (df['underdog'] == und)]
            if row.empty:
                # Try reverse matchup (shouldn't happen in real bracket)
                row = df[(df['year'] == year) & (df['round_num'] == rnd) & (df['favorite'] == und) & (df['underdog'] == fav)]
                if row.empty:
                    continue
            X = row[['favorite_probability', 'round_num', 'year']]
            p = model.predict_proba(X)[:, 1][0]
            winner = fav if np.random.rand() < p else und
            rnd_winners.append(winner)
            bracket.append({'round': rnd, 'matchup': (fav, und), 'winner': winner, 'prob': p})
        winners[rnd] = rnd_winners
    return bracket, winners

# Score a bracket
def score_bracket(bracket):
    score = 0
    for pick in bracket:
        pts = ROUND_POINTS.get(pick['round'], 0)
        score += pts
    return score

# Main simulation loop
def main(year=2019, n_sim=10000):
    df = pd.read_csv(FEATURES_CSV)
    model = joblib.load(CALIBRATED_MODEL_OUT)
    all_brackets = []
    scores = []
    for i in range(n_sim):
        bracket, _ = simulate_bracket(df, model, year)
        all_brackets.append(bracket)
        scores.append(score_bracket(bracket))
        if i % 1000 == 0:
            print(f"Simulated {i} brackets")
    print(f"Mean bracket score: {np.mean(scores):.2f}")
    print(f"Top score: {np.max(scores)}")
    # Save results
    pd.DataFrame({'score': scores}).to_csv(os.path.join(DATA_DIR, f'simulation_scores_{year}.csv'), index=False)

if __name__ == '__main__':
    main()
