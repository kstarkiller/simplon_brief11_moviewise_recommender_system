import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split


def csv_to_df(csv_path):
    # Convert csv to dataframe
    df = pd.read_csv(csv_path)
    return df


def partition(df, test_size=0.2, mini_size=0.03):
    # Partition the dataset into a training set and a test set
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)

    # Reduce the size of the training set and the test set for better performance
    train_mini = train_data[:int(mini_size * len(train_data))]
    test_mini = test_data[:int(mini_size * len(test_data))]

    return train_data, test_data, train_mini, test_mini


def NMF_training(n_components, max_iter, df):
    # Pivot train dataframe to get a matrix of users and their ratings for movies
    matrix = df.pivot(index='user_id', columns='movie_id', values='rating')

    # Fill NaN values with 
    matrix = matrix.fillna(0)

    # Drop lines with only zeros
    matrix = matrix[matrix.sum(axis=1) > 0]

    # Sparse ratings train dataframe
    matrix_sparse = matrix.astype(pd.SparseDtype("float", 0))

    # Create the model
    model = NMF(n_components=n_components, max_iter=max_iter)

    # Fit the model to the user-item train matrix
    U = model.fit_transform(matrix_sparse)  # User matrix train
    M = model.components_  # Item matrix

    pred = np.dot(U, M)
    pred_matrix = pd.DataFrame(pred, columns=matrix.columns, index=matrix.index)

    # Convert the prediction matrix to a dataframe
    pred_df = pred_matrix.stack().reset_index()
    pred_df.columns = ['user_id', 'movie_id', 'user_movie_position'] # Rename columns

    return model, pred_matrix, pred_df