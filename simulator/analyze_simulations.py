# Analyze bracket simulation results: most common paths, top brackets, risk variants
import pandas as pd
import os
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
# Find latest simulation_scores CSV
def get_latest_scores_csv():
    files = [f for f in os.listdir(DATA_DIR) if f.startswith('simulation_scores_') and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError('No simulation_scores CSV found.')
    years = [int(f.split('_')[-1].replace('.csv','')) for f in files]
    latest_year = max(years)
    return os.path.join(DATA_DIR, f'simulation_scores_{latest_year}.csv'), latest_year

SCORES_CSV, year = get_latest_scores_csv()

df = pd.read_csv(SCORES_CSV)

# Top 3 highest-scoring brackets
print('Top 3 bracket scores:')
print(df.sort_values('score', ascending=False).head(3))

# Score distribution
print('Score distribution:')
print(df['score'].describe())

# (Stub) Export top bracket as JSON/CSV (requires full bracket details from simulation engine)
# Placeholder: export scores only

df.sort_values('score', ascending=False).head(1).to_json(os.path.join(DATA_DIR, f'top_bracket_{year}.json'), orient='records')
df.sort_values('score', ascending=False).head(1).to_csv(os.path.join(DATA_DIR, f'top_bracket_{year}.csv'), index=False)

print('Analysis complete. For full path and risk variants, see simulation engine extension.')
