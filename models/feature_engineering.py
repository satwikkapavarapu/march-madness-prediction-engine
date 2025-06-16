# Feature engineering for NCAA bracket prediction
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
INPUT_CSV = os.path.join(DATA_DIR, 'historical_538_ncaa_tournament_model_results.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, 'model_features.csv')

def main():
    df = pd.read_csv(INPUT_CSV)
    # Add seed diff placeholder (to be filled if seeds available)
    df['seed_diff'] = None
    # Add KenPom feature stubs
    df['fav_adjoe'] = None
    df['fav_adjde'] = None
    df['und_adjoe'] = None
    df['und_adjde'] = None
    # Add betting odds stub
    df['fav_vegas_spread'] = None
    # Add round as integer
    df['round_num'] = pd.to_numeric(df['round'], errors='coerce')
    # Add year as integer
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    # Save engineered features
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Feature-engineered data saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
