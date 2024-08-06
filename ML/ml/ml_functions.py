import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# import required libraries
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers



### AUTOENCODER ###
def autoencoder(data, metrics=None, latent_dim=3, epochs=500):
    """
    Perform autoencoder training on a dataset.

    Parameters:
    data (pandas.DataFrame or numpy.ndarray): Input data, either a pandas DataFrame or numpy array.
    metrics (list, optional): List of column names to be used if data is a DataFrame. Ignored if data is a numpy array.
    latent_dim (int, optional): Dimension of the latent space. Default is 3.
    epochs (int, optional): Number of epochs for training. Default is 500.

    Returns:
    model_history: History of model training.
    encoded_x_train: Encoded training data.
    """
    # Check if data is a DataFrame or a numpy array
    if isinstance(data, pd.DataFrame):
        if metrics is None:
            raise ValueError("If data is a DataFrame, metrics must be provided.")
        working_data = data[metrics].to_numpy()
    elif isinstance(data, np.ndarray):
        working_data = data
    else:
        raise ValueError("Data must be either a pandas DataFrame or a numpy array.")

    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(working_data)
    x_train = scaler.transform(working_data)

    input_dim = x_train.shape[1]

    # Define the encoder model
    encoder = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(latent_dim, activation='relu')
    ])

    # Define the decoder model
    decoder = Sequential([
        Dense(64, activation='relu', input_shape=(latent_dim,)),
        Dense(128, activation='relu'),
        Dense(256, activation='relu'),
        Dense(input_dim, activation=None)
    ])

    # Combine encoder and decoder into an autoencoder model
    autoencoder = Model(inputs=encoder.input, outputs=decoder(encoder.output))
    autoencoder.compile(loss='mse', optimizer='adam')

    # Train the autoencoder
    model_history = autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=32, verbose=0)

    # Plot loss vs. epochs
    plt.plot(model_history.history["loss"])
    plt.title("Loss vs. Epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.grid(True)
    plt.show()

    # Plot original vs. reconstructed data
    def plot_orig_vs_recon(title='', n_samples=3):
        fig = plt.figure(figsize=(10, 6))
        plt.suptitle(title)
        for i in range(n_samples):
            plt.subplot(n_samples, 1, i + 1)
            idx = random.sample(range(x_train.shape[0]), 1)
            plt.plot(autoencoder.predict(x_train[idx]).squeeze(), c='steelblue', label='reconstructed' if i == 0 else '')
            plt.plot(x_train[idx].squeeze(), c='indianred', label='original' if i == 0 else '')
            if metrics:
                fig.axes[i].set_xticklabels(metrics)
            plt.xticks(np.arange(0, input_dim, 1))
            plt.grid(True)
            if i == 0:
                plt.legend()

    plot_orig_vs_recon('After training the encoder-decoder')

    # Get the encoded training data
    encoded_x_train = encoder.predict(x_train)

    return model_history, encoded_x_train


### NEURAL NETWORK ###
def neural_net(data, encoded_x_train, target_features, latent_dim=3, epochs=500):
    """
    Perform neural network prediction.

    Parameters:
    data (pd.DataFrame or np.ndarray): Original data used in the autoencoder step.
    encoded_x_train (np.ndarray): Encoded training data from the autoencoder.
    target_features (list of str): List of target features to predict.
    latent_dim (int, optional): Dimension of the latent space. Default is 3.
    epochs (int, optional): Number of epochs for training. Default is 500.

    Returns:
    model_history: History of model training.
    predicted_values: Predicted values for the test set.
    output_test: Actual values for the test set.
    """
    arr = np.array(encoded_x_train)
    # Convert the dimensions/features of latent space into a DataFrame
    ls = pd.DataFrame(arr, columns=[f'Feat{i+1}' for i in range(latent_dim)])

    # Add target variables to the latent space DataFrame
    for feature in target_features:
        ls[feature] = data[feature].values

    # Create separate train and test splits for the latent space DataFrame
    train_df, test_df = train_test_split(ls, test_size=0.2, random_state=42)

    xinput_train = train_df.iloc[:, :latent_dim].values  # Training data input invariants.
    output_train = train_df[target_features].values  # Training data target variables.
    input_test = test_df.iloc[:, :latent_dim].values  # Test data input invariants.
    output_test = test_df[target_features].values  # Test data target variables.

    # Define the neural network model
    model = Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(100, activation="elu"),
        layers.Dense(100, activation="relu"),
        layers.Dense(100, activation="elu"),
        layers.Dense(100, activation="relu"),
        layers.Dense(100, activation="relu"),
        layers.Dense(len(target_features))  # Number of output features
    ])

    model.compile(optimizer=tf.optimizers.Adam(), loss='mean_squared_error')

    # Train the neural network
    model_history = model.fit(xinput_train, output_train, epochs=epochs, validation_split=0.2, verbose=1)

    # Predict the values for the test set
    predicted_values = model.predict(input_test)

    return model_history, predicted_values, output_test


### EVALUATION METRICS ###
def evaluate_predictions(output_test, predicted_values):
    """
    Evaluate the predictions using common regression metrics.

    Parameters:
    output_test (np.ndarray): Actual values.
    predicted_values (np.ndarray): Predicted values.

    Returns:
    dict: Dictionary containing MSE, RMSE, MAE, and R².
    """
    mse = mean_squared_error(output_test, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(output_test, predicted_values)
    r2 = r2_score(output_test, predicted_values)

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}


def compare_lists(list_a,list_b):
  ''' 
  This function takes as inputs two lists or numpy arrays of the same size, and returns 
  the percentage of how many rows match between them.
  '''
  array_a=np.array(list_a)
  array_b=np.array(list_b)
  if (array_a).shape!=array_b.shape:
    print("Error: Input lists are not the same shapes.")
    return
  if array_a.ndim==1:
    array_a=array_a.reshape(len(array_a),1)
    array_b=array_b.reshape(len(array_b),1)
  array_a=np.around(array_a)
  array_b=np.around(array_b)
  return np.sum([ all(x==y) for (x,y) in zip(array_a, array_b)])/len(array_a)
