import os 
import csv
import pandas as pd 


def text_to_csv(src_path, dest_path, filename):
    """Reads the text file containing the paths to images
       to a csv file. """ 

    """
    Args:
        src_path(string): path to the txt file. 
        dest_path(string): path to the directory to store the csv file. 
        filename(string): name of the csv file.
    """
    master_list = [] 
    if filename == "train.csv":
        header_list = ["color", "depth", "ir", "label"]
    elif filename == "valid.csv":
        header_list = ["color", "depth", "ir"]
    else: 
        print("[Filename Error] The filename can be either train.csv or valid.csv")
        return 
    
    with open(src_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            split_line = line.split(' ')
            master_list.append(split_line)

    dest_path = os.path.join(dest_path, filename)

    with open(dest_path, 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_list)
        writer.writerows(master_list)



if __name__ == "__main__":

    # Processing the train.txt file 
    src_path = r"F:\Projects\PersonalProjects-GitHub\facespoof-detection\data\train_list.txt"
    dest_path = os.path.dirname(src_path)
    filename = "train.csv"
    text_to_csv(src_path, dest_path, filename)


    #Preprocessing the valid.txt file
    src_path = r"F:\Projects\PersonalProjects-GitHub\facespoof-detection\data\val_public_list.txt"
    dest_path = os.path.dirname(src_path)
    filename = "valid.csv"
    text_to_csv(src_path, dest_path, filename)


    # Converting unix path to windows path - train.csv
    input_path = r"F:\Projects\PersonalProjects-GitHub\facespoof-detection\data"
    df = pd.read_csv(r"F:\Projects\PersonalProjects-GitHub\facespoof-detection\data\train.csv")

    # Converting unix path to win path
    df['color'] = df['color'].apply(lambda x: os.path.normpath(x))
    df['depth'] = df['depth'].apply(lambda x: os.path.normpath(x))
    df['ir'] = df['ir'].apply(lambda x: os.path.normpath(x))
    df.to_csv(os.path.join(input_path, "train.csv"), index=False)
    
    # Converting unix path to windows path - valid.csv
    input_path = r"F:\Projects\PersonalProjects-GitHub\facespoof-detection\data"
    df = pd.read_csv(r"F:\Projects\PersonalProjects-GitHub\facespoof-detection\data\valid.csv")

    # Converting unix path to win path
    df['color'] = df['color'].apply(lambda x: os.path.normpath(x))
    df['depth'] = df['depth'].apply(lambda x: os.path.normpath(x))
    df['ir'] = df['ir'].apply(lambda x: os.path.normpath(x))
    
    df.to_csv(os.path.join(input_path, "valid.csv"), index=False)
