import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder

if __name__ == "__main__":
    csv_file = "20250704_nonull.csv"
    csv_data = pd.read_csv(csv_file)
    ignored_columns = ['pitch_type', 'game_date', 'spin_dir', 'spin_rate_deprecated', 'break_angle_deprecated', 'break_length_deprecated', 
                       'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire', 'sv_id', 'player_name', 'des', 'batter', 'pitcher', 'events',
                       'home_team', 'away_team', 'game_year', 'pitch_name']
    X_train = csv_data.drop(columns=ignored_columns)
    X_train = OneHotEncoder().fit_transform(X_train)
    y_train = csv_data['pitch_type']
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)