import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

if __name__ == "__main__":
    csv_file = "20250704_nonull.csv"
    ignored_columns = ['pitch_type', 'game_date', 'spin_dir', 'spin_rate_deprecated', 'break_angle_deprecated', 'break_length_deprecated', 
                       'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire', 'sv_id', 'player_name', 'des', 'events',
                       'home_team', 'away_team', 'game_year', 'pitch_name', 'hit_location', 'bb_type', 'on_3b', 'on_2b', 'on_1b', 'hc_x',
                       'hc_y', 'hit_distance_sc', 'launch_speed', 'launch_angle', 'game_pk', 'fielder_2', 'fielder_3', 'fielder_4',
                       'fielder_5', 'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9', 'estimated_ba_using_speedangle',
                       'estimated_woba_using_speedangle', 'woba_value', 'woba_denom', 'babip_value', 'iso_value', 'launch_speed_angle',
                       'bat_speed', 'swing_length', 'estimated_slg_using_speedangle', 'age_pit_legacy', 'age_bat_legacy', 'attack_angle',
                       'attack_direction', 'swing_path_tilt', 'intercept_ball_minus_batter_pos_x_inches', 'intercept_ball_minus_batter_pos_y_inches']
    
    # Set up X and y data using the CSV file.
    csv_data = pd.read_csv(csv_file, encoding_errors='ignore')
    X = csv_data.drop(columns=ignored_columns)
    X = OneHotEncoder().fit_transform(X)
    y = csv_data['pitch_type']
    
    # Split the X and y data into training data and validation data (80/20).
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Train and evaluate model.
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"Accuracy of Model: {accuracy:.4f}")