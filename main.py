import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

if __name__ == "__main__":
    # csv_train_file = "train.csv"
    # csv_validation_file = "validate.csv"
    csv_file = "20250704_nonull.csv"
    ignored_columns = ['pitch_type', 'game_date', 'spin_dir', 'spin_rate_deprecated', 'break_angle_deprecated', 'break_length_deprecated', 
                       'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire', 'sv_id', 'player_name', 'des', 'batter', 'pitcher', 'events',
                       'home_team', 'away_team', 'game_year', 'pitch_name']
    
    # Set up training data.
    # csv_train_data = pd.read_csv(csv_train_file, encoding_errors='ignore')
    # X_train = csv_train_data.drop(columns=ignored_columns)
    # X_train = OneHotEncoder().fit_transform(X_train)
    # y_train = csv_train_data['pitch_type']
    
    # Set up validation data.
    # csv_validation_data = pd.read_csv(csv_validation_file, encoding_errors='ignore')
    # X_val = csv_validation_data.drop(columns=ignored_columns)
    # X_val = OneHotEncoder().fit_transform(X_val)
    # y_val = csv_validation_data['pitch_type']

    csv_data = pd.read_csv(csv_file, encoding_errors='ignore')
    X = csv_data.drop(columns=ignored_columns)
    X = OneHotEncoder().fit_transform(X)
    y = csv_data['pitch_type']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Train and evaluate model.
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"Accuracy of Model: {accuracy:.4f}")