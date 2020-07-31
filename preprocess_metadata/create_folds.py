import os 
import pandas as pd 
from sklearn import preprocessing 
from sklearn import model_selection


if __name__ == "__main__":
    input_path = r"F:\Projects\PersonalProjects-GitHub\facespoof-detection\data"
    df = pd.read_csv(os.path.join(input_path, "train_rgb.csv"))
    df.kfold = -1 

    df = df.sample(frac=1).reset_index(drop = True)
    y = df.label.values
    kf = model_selection.StratifiedKFold(n_splits = 10)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_idx, 'kfold'] = fold

    df.to_csv(os.path.join(input_path, "train_folds.csv"), index=False)

