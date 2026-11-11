import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def detect_feature_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, threshold: float = 0.05):
    drift_report = {}
    for col in reference_data.columns:
        if col in current_data.columns:
            stat, p_value = ks_2samp(reference_data[col], current_data[col])
            drift_report[col] = {
                'p_value': round(p_value, 4),
                'drift_detected': p_value < threshold
            }
    drifted_cols = [k for k, v in drift_report.items() if v['drift_detected']]
    print(f'Drift Check Complete. Drift detected in {len(drifted_cols)} feature(s): {drifted_cols}')
    return drift_report

if __name__ == '__main__':
    ref_df = pd.read_csv('data/data.csv').drop(columns=['id', 'diagnosis', 'Unnamed: 32'], errors='ignore')
    curr_df = ref_df.copy() * np.random.uniform(0.95, 1.05, size=ref_df.shape)
    detect_feature_drift(ref_df, curr_df)
