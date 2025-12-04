import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_experiment(csv_file: str):
    # Selecting a small set of columns
    numeric_cols = [
        "release_speed",
        "release_pos_x",
        "release_pos_z",
        "pfx_x",
        "pfx_z",
        "plate_x",
        "plate_z",
        "balls",
        "strikes",
        "sz_top",
        "sz_bot",
        "launch_speed",
        "launch_angle",
    ]

    cat_cols = [
        "stand",
        "p_throws",
        "inning_topbot",
        "zone",
        "game_type",
        "type",     # ball/strike/in-play
        "bb_type",
    ]

    use_cols = numeric_cols + cat_cols + ["pitch_type"]

    # Read only the columns we want
    csv_data = pd.read_csv(csv_file, encoding_errors="ignore", usecols=use_cols)
    print(f"Loaded {csv_file} with shape: {csv_data.shape}")
    
    # if csv_file == "train.csv":
    #    n_sample = min(5000, len(csv_data))
    #    csv_data = csv_data.sample(n=n_sample, random_state=42)
    #    print(f"Using a {n_sample}-row sample from {csv_file}")

    # drop rows without labels
    csv_data = csv_data.dropna(subset=["pitch_type"])

    # Handle missing values in features
    # fill NaNs with column medians
    csv_data[numeric_cols] = csv_data[numeric_cols].fillna(csv_data[numeric_cols].median())

    # fill NaNs with "Unknown"
    csv_data[cat_cols] = csv_data[cat_cols].fillna("Unknown")

    # drop pitch types that appear only once (can't stratify on them)
    counts = csv_data["pitch_type"].value_counts()
    rare_types = counts[counts < 2].index
    if len(rare_types) > 0:
        print("Dropping rare pitch types with < 2 samples:", list(rare_types))
        csv_data = csv_data[~csv_data["pitch_type"].isin(rare_types)]

    # build x and y
    X_num = csv_data[numeric_cols].copy()
    X_cat = csv_data[cat_cols].copy()
    y = csv_data["pitch_type"]

    # Scale numeric features
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # One hot encode categorical features
    X_cat_dummies = pd.get_dummies(X_cat)

    # combine into one feature
    X = np.hstack([X_num_scaled, X_cat_dummies.values])
    print("X shape after scaling and encoding:", X.shape)

    # Train test split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Tuned Logistic Regression
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",  # helps if some pitch types are rare
        n_jobs=-1,                
        solver="lbfgs",
        C=2.0 # default is 1.0, higher C = less regularization
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"Accuracy of Model: {accuracy:.4f}")


if __name__ == "__main__":
    # Use provided file, else default to 20250704_nonull.csv
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        print(f"Using csv file: {csv_file}")
    else:
        csv_file = "20250704_nonull.csv"
        print("No csv provided, using default: 20250704_nonull.csv")

    run_experiment(csv_file)
