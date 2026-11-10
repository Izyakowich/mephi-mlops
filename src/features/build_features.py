import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1. Биннинг возраста
    df['AGE_BIN'] = pd.cut(df['AGE'], bins=[0, 25, 35, 45, 55, 100], labels=[0, 1, 2, 3, 4]).astype(int)
    
    # 2. Агрегации по платежам
    pay_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    df['PAY_AMT_MEAN'] = df[pay_cols].mean(axis=1)
    df['PAY_AMT_SUM'] = df[pay_cols].sum(axis=1)
    
    # 3. Отношение платежа к счету
    df['PAY_TO_BILL_RATIO'] = df['PAY_AMT1'] / (df['BILL_AMT1'] + 1)
    
    return df