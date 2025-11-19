import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

if __name__ == "__main__":
    csv_file = "20250704.csv"
    csv_data = pd.read_csv(csv_file)
    X_train = csv_data.drop(columns=['pitch_type'])
    y_train = csv_data['pitch_type']
    model = LogisticRegression().fit(X_train, y_train)