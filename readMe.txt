Run the model with:
python model.py train.csv f

The first command line argument is the file containing the data to use we are
including train.csv in our zip file.

The second command like argument optionally DISABLES one-hot encoding using
player ids. It is on by default, but we recommend disabling it.

The model should run relatively quickly and print out a balanced accuracy score,
F1 score, the top 5 features for each class, and a confusion matric to the
terminal.