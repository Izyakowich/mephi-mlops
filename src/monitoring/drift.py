import pandas as pd
import numpy as np
import joblib
import sys
sys.path.append(".")

def calculate_psi(expected, actual, buckets=10):
    """Population Stability Index"""
    def scale_range(input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = scale_range(breakpoints, np.min(expected), np.max(expected))
    
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    def sub_psi(e_perc, a_perc):
        if e_perc == 0: e_perc = 0.0001
        if a_perc == 0: a_perc = 0.0001
        return (e_perc - a_perc) * np.log(e_perc / a_perc)

    return np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(len(expected_percents)))

if __name__ == "__main__":
    model = joblib.load("models/credit_default_model.pkl")
    
    # Имитируем новые данные (берем тестовую выборку)
    train = pd.read_csv("data/processed/train.csv").drop(columns=['target'])
    test = pd.read_csv("data/processed/test.csv").drop(columns=['target'])
    
    train_probs = model.predict_proba(train)[:, 1]
    test_probs = model.predict_proba(test)[:, 1]
    
    psi = calculate_psi(train_probs, test_probs)
    print(f"PSI: {psi:.4f}")
    if psi > 0.25:
        print("WARNING: Significant Data Drift Detected!")
    else:
        print("Data is stable.")