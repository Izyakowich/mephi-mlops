import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split
import sys
sys.path.append(".")
from src.features.build_features import build_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/UCI_Credit_Card.csv")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    df = pd.read_csv(args.input)
    df = df.rename(columns={"default.payment.next.month": "target"})
    df = df.drop(columns=["ID"])
    
    # Feature Engineering
    df = build_features(df)
    
    # Train / Test Split
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    
    train.to_csv(f"{args.out_dir}/train.csv", index=False)
    test.to_csv(f"{args.out_dir}/test.csv", index=False)
    print(f"Data saved to {args.out_dir}")

if __name__ == "__main__":
    main()