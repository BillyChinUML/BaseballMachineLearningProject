import os, sys
import numpy as np
import pandas as pd

#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

def reclassify_pitches(df):
        # Mapping Statcast pitch_type to our three categories
    mapping = {
        'FF': 'F', # four-seam fastball
        'SI': 'F', # sinker
        'FC': 'F', # cutter
        'FS': 'F', # splitter
        'FO': 'F', # forkball
        'FA': 'F', # two-steam fastball
        'CU': 'B', # curveball
        'CS': 'B', # slow curveball
        'SL': 'B', # slider
        'SC': 'B', # screwball
        'ST': 'B', # sweeper
        'SV': 'B', # slurve
        'KC': 'B', # knucklecurve
        'CH': 'O', # changeup
        'KN': 'O', # knuckleball
    }
    
    # Apply the mapping, replace values not in the mapping with NaN
    df['pitch_type'] = df['pitch_type'].map(mapping).fillna(df['pitch_type'])

    df = df[~df['pitch_type'].isin(['EP', 'PO', 'UN'])]

    return df


# def add_velo_diff_feature(df):
    # hardest pitch - release_speed
    # average release_speed - release_speed
    # break vs differences in break (maybe can use bins for this?)
    # (this all needs to be unique to each player too)


def process_csv(file):
    file_stem, file_extension = os.path.splitext(file)
    cols = [
        'pitch_type',
        'release_speed',
        'release_pos_x',
        'release_pos_z',
        'pitcher',
        'p_throws',
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
    df = pd.read_csv(file)
    df = df[cols]
    df = reclassify_pitches(df)

    df.to_csv(file_stem + '_processed' + file_extension, index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file = sys.argv[1]
        print(f"Processing csv file: {file}")
    else:
        file = "alldata.csv"
        print(f"Processing default csv file: {file}")
    process_csv(file)

# cols = ['pitch_type', 'game_date', 'release_speed', 'release_pos_x', 'release_pos_z', 'player_name', 'batter', 'pitcher', 'events', 'description', 'spin_dir', 'spin_rate_deprecated', 'break_angle_deprecated', 'break_length_deprecated', 'zone', 'des', 'game_type', 'stand', 'p_throws', 'home_team', 'away_team', 'type', 'hit_location', 'bb_type', 'balls', 'strikes', 'game_year', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'inning_topbot', 'hc_x', 'hc_y', 'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire', 'sv_id', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top', 'sz_bot', 'hit_distance_sc', 'launch_speed', 'launch_angle', 'effective_speed', 'release_spin_rate', 'release_extension', 'game_pk', 'fielder_2', 'fielder_3', 'fielder_4', 'fielder_5', 'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9', 'release_pos_y', 'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle', 'woba_value', 'woba_denom', 'babip_value', 'iso_value', 'launch_speed_angle', 'at_bat_number', 'pitch_number', 'pitch_name', 'home_score', 'away_score', 'bat_score', 'fld_score', 'post_away_score', 'post_home_score', 'post_bat_score', 'post_fld_score', 'if_fielding_alignment', 'of_fielding_alignment', 'spin_axis', 'delta_home_win_exp', 'delta_run_exp', 'bat_speed', 'swing_length', 'estimated_slg_using_speedangle', 'delta_pitcher_run_exp', 'hyper_speed', 'home_score_diff', 'bat_score_diff', 'home_win_exp', 'bat_win_exp', 'age_pit_legacy', 'age_bat_legacy', 'age_pit', 'age_bat', 'n_thruorder_pitcher', 'n_priorpa_thisgame_player_at_bat', 'pitcher_days_since_prev_game', 'batter_days_since_prev_game', 'pitcher_days_until_next_game', 'batter_days_until_next_game', 'api_break_z_with_gravity', 'api_break_x_arm', 'api_break_x_batter_in', 'arm_angle', 'attack_angle', 'attack_direction', 'swing_path_tilt', 'intercept_ball_minus_batter_pos_x_inches', 'intercept_ball_minus_batter_pos_y_inches']



# Potential improvement: train with data set of starters, limit number of unique pitchers
