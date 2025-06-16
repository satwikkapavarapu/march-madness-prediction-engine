import streamlit as st
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
FEATURES_CSV = os.path.join(DATA_DIR, 'model_features.csv')
SHAP_CSV = os.path.join(DATA_DIR, 'shap_feature_importance.csv')
TOP_BRACKET_JSON = None
for f in os.listdir(DATA_DIR):
    if f.startswith('top_bracket_') and f.endswith('.json'):
        TOP_BRACKET_JSON = os.path.join(DATA_DIR, f)
        break

st.title('March Madness Bracket Prediction Engine')
st.markdown('''A production-grade NCAA tournament bracket prediction engine using advanced ML, probability calibration, and full-tournament simulation.\n\n**Features:**\n- Simulated brackets\n- SHAP explainability\n- Export results (JSON/CSV)''')

# Show SHAP feature importance if available
if os.path.exists(SHAP_CSV):
    shap_df = pd.read_csv(SHAP_CSV)
    st.subheader('Feature Importance (SHAP)')
    st.dataframe(shap_df)

# Show top bracket (stub: currently only score)
if TOP_BRACKET_JSON and os.path.exists(TOP_BRACKET_JSON):
    st.subheader('Top Simulated Bracket (Score Only Stub)')
    top_df = pd.read_json(TOP_BRACKET_JSON)
    st.dataframe(top_df)
    st.download_button('Download Top Bracket JSON', data=top_df.to_json(orient='records'), file_name='top_bracket.json')
    st.download_button('Download Top Bracket CSV', data=top_df.to_csv(index=False), file_name='top_bracket.csv')
else:
    st.info('No top bracket results available yet. Run simulation and analysis.')

# User controls for future extension (risk preference, custom weights)
st.sidebar.header('Simulation Controls (Coming Soon)')
st.sidebar.slider('Risk Preference', min_value=0, max_value=100, value=50)
st.sidebar.text('Custom weights and pool rules coming soon!')
