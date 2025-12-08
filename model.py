import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

#pd.set_option('display.max_columns', None)

def add_quadratic_features(df, numeric_cols, deg=2):
    poly = PolynomialFeatures(degree=deg, include_bias=False)

    df_X = df[numeric_cols]
    df_other = df.drop(columns=numeric_cols)
    
    X_poly = poly.fit_transform(df_X)
    
    # Create new column names for the quadratic features
    feature_names = poly.get_feature_names_out(input_features=list(df_X.columns))
    
    # Create a DataFrame for the new quadratic features
    df_poly = pd.DataFrame(X_poly, columns=feature_names)
    # Concatenate the original DataFrame with the new quadratic features
    df_new = pd.concat([df_other, df_poly], axis=1)
    
    return df_new


def run_experiment(csv_file: str, one_hot_encoding=True):
    numeric_cols = [
        'release_speed',
        'release_pos_x',
        'release_pos_z',
        'pfx_x',
        'pfx_z',
        'plate_x',
        'plate_z',
        'vx0',
        'vy0',
        'vz0',
        'ax',
        'ay',
        'az',
        'effective_speed',
        'release_extension',
        'release_pos_y',
        'spin_axis'
    ]
    
    cat_cols = [
        'pitcher',
        'p_throws',
    ]

    # Read only the columns we want
    df = pd.read_csv(csv_file, encoding_errors='ignore')
    print(f"Loaded {csv_file} with shape: {df.shape}")
    
    # if csv_file == "train.csv":
    #    n_sample = min(5000, len(csv_data))
    #    csv_data = csv_data.sample(n=n_sample, random_state=42)
    #    print(f"Using a {n_sample}-row sample from {csv_file}")

    # drop rows without labels
    df = df.dropna(subset=['pitch_type'])

    # Handle missing values in features
    # fill NaNs with column medians
    
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # fill NaNs with "Unknown"
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    df = add_quadratic_features(df, numeric_cols)

    # drop pitch types that appear only once (can't stratify on them)
    counts = df["pitch_type"].value_counts()
    rare_types = counts[counts < 2].index
    if len(rare_types) > 0:
        print("Dropping rare pitch types with < 2 samples:", list(rare_types))
        df = df[~df["pitch_type"].isin(rare_types)]

    # build x and y
    X_num = df[numeric_cols].copy()
    X_cat = df[cat_cols].copy()
    X_num = df.drop(columns=['pitch_type'] + cat_cols)
 
    y = df['pitch_type']

    # Scale numeric features
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # One hot encode categorical features
    X_cat_dummies = pd.get_dummies(X_cat)
    if one_hot_encoding:
        pitcher_encoding = pd.get_dummies(X_cat['pitcher'])
        p_throw_encoding = pd.get_dummies(X_cat['p_throws'])
        X_cat_dummies = pd.concat([pitcher_encoding, p_throw_encoding], axis=1)

    numeric_column_names = X_num.columns

    cat_column_names = cat_cols
    if one_hot_encoding:
        cat_column_names = list(pitcher_encoding.columns) + list(p_throw_encoding.columns)
    feature_names = np.concatenate([numeric_column_names, cat_column_names])

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
        C=0.5 # default is 1.0, higher C = less regularization
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    bal_accuracy = balanced_accuracy_score(y_val, y_pred)
    matrix = confusion_matrix(y_val, y_pred)

    print(f"Accuracy of model: {accuracy:.4f}")
    print(f"Balanced accuracy of model: {bal_accuracy:.4f}")

    print("Confusion Matrix:")
    print(matrix)

    # This part is to try and determine feature importance
    coefficients = model.coef_
    for class_idx, class_coef in enumerate(coefficients):
        print(f"\nMost positive coefficients for Class {class_idx}:")
        
        top_positive_indices = np.argsort(class_coef)[-10:]

        for idx in top_positive_indices:
            feature_name = feature_names[idx]
            print(f"  {feature_name}: {class_coef[idx]:.4f}")

        # print(f"\nMost negative coefficients for Class {class_idx}:")

        # top_negative_indices = np.argsort(class_coef)[:10]

        # for idx in top_negative_indices:
        #     feature_name = feature_names[idx]
        #     print(f"  {feature_name}: {class_coef[idx]:.4f}")


if __name__ == "__main__":
    # Use provided file, else default to 20250704_nonull.csv
    one_hot_encoding = True
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        print(f"Using csv file: {csv_file}")
        if sys.argv[2] == 'f':
            one_hot_encoding = False

    else:
        csv_file = "20250704_nonull.csv"
        #csv_file = 'data.csv'
        print(f"No csv provided, using default: {csv_file}")
    run_experiment(csv_file, one_hot_encoding)
