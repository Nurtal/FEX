import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def extract_features(data_file:str) -> dict:
    """Run Linear Discriminant analysis to extract important features from data_file.
    LDA is run using cross validation, feature importance as computed as the mean of absolute value of
    each contribution accross the cross validation. 
    
    Args:
        - data_file (str) : path to the data file, must contains a 'LABEL' columns

    Returns:
        - (dict) : feature to absolute value of contribution
        
    """

    # load data
    df = pd.read_csv(data_file)
    X = df.drop(columns=["LABEL"])
    nb = 0
    label_to_nb = {}
    for label in set(df['LABEL']):
        label_to_nb[label] = nb
        nb+=1
    y = df["LABEL"].map(label_to_nb)

    # cross validation split
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    coefficients_list = []

    # train on each fold
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
    
        # extract coeff
        coefficients_list.append(abs(lda.coef_))

    # compute mean of coefs
    mean_coefficients = np.mean(coefficients_list, axis=0)

    # sort and save
    feature_to_coef = {}
    for feature_name, coef in zip(list(X.keys()), mean_coefficients[0]):
        feature_to_coef[feature_name] = abs(coef)
    feature_to_coef = dict(sorted(feature_to_coef.items(), key=lambda item: item[1], reverse=True))

    # return selected features and their contribution
    return feature_to_coef



if __name__ == "__main__":

    data_file = "data/toy_data.csv" 
    # m = extract_features(data_file)
    # print(m)

    # Bene stuff
    data_file_list = [
        "/home/bran/Workspace/SIDEQUEST/Bene/thesis/feature_extraction/data/rnaseq_c3_vs_all.csv",
        "/home/bran/Workspace/SIDEQUEST/Bene/thesis/feature_extraction/data/rnaseq_c3_vs_clust1.csv",
        "/home/bran/Workspace/SIDEQUEST/Bene/thesis/feature_extraction/data/rnaseq_c3_vs_clust2.csv"
    ]
    for tf in data_file_list:
        feature_to_coef = extract_features(tf)
        save_file = open(tf.replace("/data/", "/fex/").replace(".csv", "_lda_feature_extraction.csv"), "w")
        save_file.write("FEATURE,IMPORTANCE\n")
        for k in feature_to_coef:
            save_file.write(f"{k},{feature_to_coef[k]}\n")
        save_file.close()
    
