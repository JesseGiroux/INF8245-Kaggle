import os
import copy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score as compute_f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit, ParameterGrid



# Useful constants
EPSILON = 10e-9
PRECISION = 3
SCORING = 'f1_macro'

AXIS_ROW = 1 # NumPy
AXIS_COL = 0
AXIS_EXAMPLE = 'index' # Pandas
AXIS_FEATURE = 'columns'

RESOLUTION_X = 96
RESOLUTION_Y = 96
RESOLUTION = (RESOLUTION_X, RESOLUTION_Y)
N_PIXELS = RESOLUTION[0] * RESOLUTION[-1]
PIXEL_MIN = 0
PIXEL_MAX = 255

# Datasets
SPLITS = ['Train', 'Valid', 'Test']
SPLIT_DATA = { 'x': None, 'y': None }
TRAIN_RATIO = 0.8

CLASS_LABELS = ['big_cats', 'butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'goat', 'horse', 'spider', 'squirrel']
N_CLASSES = len(CLASS_LABELS)

DATASETS_PATH = './Datasets'
RESULTS_PATH = './Results'



# ************************************************** DATA R/W ************************************************** #
def load_pkl(path):
    pkl = None

    with open(path, 'rb') as f:
        pkl = pickle.load(f)

    return pkl



def load_data():
    data = { }

    # Read PKL files
    for split in SPLITS:
        data[split] = copy.deepcopy(SPLIT_DATA)

        for label in SPLIT_DATA.keys():
            path = f'{DATASETS_PATH}/{label.lower()}_{split.lower()}.pkl'

            if os.path.exists(path):
                print(f'Loading data: {label.upper()} ({split})')
                
                data[split][label] = np.array(load_pkl(path))

                # Scale pixel values (0-255 -> 0-1)
                if label == 'x':
                    x = data[split]['x']

                    x = np.array(x, dtype = float) / PIXEL_MAX

                    data[split]['x'] = x

    return data



def split_data(data, ratio):

    '''
    Splits dataset into two according to given ratio, while keeping order.
    '''

    # Compute dataset size
    size = len(data)

    # Split data into two random sets
    split_size = int(round(ratio * size))

    random_indices = np.arange(size, dtype = int)
    np.random.shuffle(random_indices)

    a = data[random_indices[:split_size]]
    b = data[random_indices[split_size:]]

    return [a, b]






# ************************************************** PLOTTING ************************************************** #
def plot_image(img, title = '...'):

    # Initialize plot
    _, ax = plt.subplots(figsize = (10, 8))

    # Define plot title and axis labels
    ax.set_title(title, fontweight = 'bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Plot image
    ax.imshow(img)

    # Tighten up the whole plot
    plt.tight_layout()

    # Show plot
    plt.show()



def plot_image_grid(imgs, labels, n_rows, n_cols):
    if ((not type(n_rows) is int or not type(n_cols) is int) or
        (n_rows <= 0 or n_cols <= 0) or
        (n_rows == 1 and n_cols == 1)):
        raise ValueError('Invalid grid.')

    if n_rows * n_cols != len(imgs):
        raise ArithmeticError("Grid doesn't fit images.")

    # Initialize plot
    _, axes = plt.subplots(n_rows, n_cols, figsize = (10, 10))

    # Add every image to grid
    for nth in range(len(imgs)):
        [i, j] = np.unravel_index(nth, (n_rows, n_cols))

        ax = axes[i, j]
        img = imgs[nth]
        label = labels[nth]

        # Set image title
        ax.set_title(label, fontweight = 'bold', fontsize = 10)

        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Plot image
        ax.imshow(img)

    # Tighten up the whole plot
    plt.tight_layout()

    # Show plot
    plt.show()






# ************************************************** DATA EXPLORATION ************************************************** #
def explore_data(x, y):
    print('Exploring data...')

    n_train = len(x)
    print(f'# examples: {n_train}')

    res_x, res_y = x[0].shape
    print(f'Image resolution: ({res_x}, {res_y})')

    # Plot random training images in a grid
    n_rows, n_cols = 5, 5
    n_images = n_rows * n_cols

    indices = list(range(n_train))
    np.random.shuffle(indices)
    indices = indices[:n_images]

    print(f'Plot {n_images} training images...')

    plot_image_grid(x[indices], y[indices], n_rows, n_cols)






# ************************************************** MODELS ************************************************** #
def train_neural_network(x_train, y_train):

    # Define neural network
    n_input = N_PIXELS
    n_hidden_layers = 9
    n_output = N_CLASSES

    # Use tower model for deep layers
    # First layer: 9216
    # Last layer: 11
    # Number of hidden layers: 9
    hidden_layer_sizes = [int(N_PIXELS / 2**(i + 1)) for i in range(n_hidden_layers)]

    # Train NN
    model = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes, activation = 'relu', verbose = 3)
    model.fit(x_train, y_train)

    return model






def main():

    # Read data
    data = load_data()

    [x_train, y_train] = [data['Train'][i] for i in ['x', 'y']]
    [x_test, _] = [data['Test'][i] for i in ['x', 'y']]


    # Split the training data into training/validation subsets
    [train_data, valid_data] = split_data(np.array(list(zip(x_train, y_train)), dtype = object), TRAIN_RATIO)

    [x_train, y_train] = [np.array(d) for d in list(zip(*train_data))]
    [x_valid, y_valid] = [np.array(d) for d in list(zip(*valid_data))]


    # Compute size of each data subset
    n_train, n_valid, n_test = [len(x) for x in [x_train, x_valid, x_test]]
    print(f'# examples (Training): {n_train}')
    print(f'# examples (Validation): {n_valid}')
    print(f'# examples (Test): {n_test}')


    # Explore dataset
    #explore_data(x_train, y_train)


    # Flatten data
    x_train = np.reshape(x_train, (n_train, N_PIXELS))
    x_valid = np.reshape(x_valid, (n_valid, N_PIXELS))
    x_test = np.reshape(x_test, (n_test, N_PIXELS))


    # Train models
    model = train_neural_network(x_train, y_train)

    # Predict
    f1_train = compute_f1_score(y_train, model.predict(x_train), average = 'macro')
    print(f'F1-Score (Training): {f1_train}')
    f1_valid = compute_f1_score(y_valid, model.predict(x_valid), average = 'macro')
    print(f'F1-Score (Validation): {f1_valid}')






if __name__ == '__main__':
    main()