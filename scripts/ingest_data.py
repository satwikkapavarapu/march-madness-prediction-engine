# Download FiveThirtyEight NCAA tournament historical results for modeling
import os
import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
FIVETHIRTYEIGHT_URL = 'https://raw.githubusercontent.com/fivethirtyeight/data/master/historical-ncaa-forecasts/historical-538-ncaa-tournament-model-results.csv'

def download_csv(url, fname):
    resp = requests.get(url)
    resp.raise_for_status()
    out_path = os.path.join(DATA_DIR, fname)
    with open(out_path, 'wb') as f:
        f.write(resp.content)
    print(f"Downloaded {fname} to {out_path}")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    download_csv(FIVETHIRTYEIGHT_URL, 'historical_538_ncaa_tournament_model_results.csv')
    print("FiveThirtyEight NCAA tournament data downloaded.")

if __name__ == '__main__':
    main()
