import os
import copy
import json
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

AXIS_ROW = 0 # NumPy
AXIS_COL = 1
AXIS_EXAMPLE = 'index' # Pandas
AXIS_FEATURE = 'columns'

RESOLUTION_X = 96
RESOLUTION_Y = 96
RESOLUTION = (RESOLUTION_X, RESOLUTION_Y)
N_PIXELS = RESOLUTION[0] * RESOLUTION[-1]
PIXEL_MIN = 0
PIXEL_MAX = 255

# Models
MAX_DECISION_TREE_DEPTH = 100

# Datasets
SPLITS = ['Train', 'Valid', 'Test']
SPLIT_DATA = { 'x': None, 'y': None }
TRAIN_RATIO = 0.8

DATASETS_PATH = './Datasets'
RESULTS_PATH = './Results'

CLASS_LABELS = ['big_cats', 'butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'goat', 'horse', 'spider', 'squirrel']
N_CLASSES = len(CLASS_LABELS)






# ************************************************** HELPER FUNCTIONS ************************************************** #
def print_f1s(f1s, name):
    for split, f1 in f1s.items():
        print(f"F1-Score {split} ({name}): {round(f1, PRECISION)}")






# ************************************************** DATA R/W ************************************************** #
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_dataframe(path, dtype = None):
    if os.path.isfile(path):
        with open(path, 'r', encoding = 'UTF-8') as f:
            return pd.read_csv(f, header = 0, index_col = False, dtype = dtype)

def load_series(path, name = None, dtype = None):
    if os.path.isfile(path):
        with open(path, 'r', encoding = 'UTF-8') as f:
            s = pd.read_csv(f, header = None, index_col = 0, squeeze = True, dtype = dtype)

            # Define series name
            s.name = name

            # Remove axis name
            s.axes[0].name = None

            return s



def store_dataframe(df, path):
    with open(path, 'w', encoding = 'UTF-8') as f:
        df.to_csv(f, index = False, header = True, line_terminator = '\n')

def store_series(s, path):
    with open(path, 'w', encoding = 'UTF-8') as f:
        s.to_csv(f, index = True, header = False, line_terminator = '\n')



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

    # Split the training data into training/validation subsets
    train_data = np.array(list(zip(data['Train']['x'], data['Train']['y'])), dtype = object)
    [train_data, valid_data] = split_data(train_data, TRAIN_RATIO)

    [x_train, y_train] = [np.array(d) for d in list(zip(*train_data))]
    [x_valid, y_valid] = [np.array(d) for d in list(zip(*valid_data))]

    data['Train']['x'] = x_train
    data['Train']['y'] = y_train
    data['Valid']['x'] = x_valid
    data['Valid']['y'] = y_valid

    return data



def split_data(data, ratio):

    '''
    Splits dataset into two according to given ratio.
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



def flatten_data(data):
    flat_data = copy.deepcopy(data)
    
    for split in data.keys():
        for label in data[split].keys():
            if label == 'x':
                x = data[split][label]

                flat_data[split][label] = np.reshape(x, (len(x), N_PIXELS))

    return flat_data






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
def explore_data(data, plot = False):
    print('Exploring data...')
    print()


    # Read data
    [x_train, y_train] = [data['Train'][i] for i in ['x', 'y']]
    [x_valid, y_valid] = [data['Valid'][i] for i in ['x', 'y']]
    [x_test, _] = [data['Test'][i] for i in ['x', 'y']]


    # Compute size of each data subset
    n_train, n_valid, n_test = [len(x) for x in [x_train, x_valid, x_test]]
    print(f'# examples (Training): {n_train}')
    print(f'# examples (Validation): {n_valid}')
    print(f'# examples (Test): {n_test}')
    print()


    # Compute distribution of classes    
    train_labels, train_counts = np.unique(y_train, return_counts = True)
    valid_labels, valid_counts = np.unique(y_valid, return_counts = True)

    train_class_distribution = pd.Series(np.zeros(N_CLASSES), index = CLASS_LABELS)
    valid_class_distribution = pd.Series(np.zeros(N_CLASSES), index = CLASS_LABELS)
    train_class_distribution.loc[train_labels] = train_counts / np.sum(train_counts)
    valid_class_distribution.loc[valid_labels] = valid_counts / np.sum(valid_counts)
    print(f'Distribution of classes (Training):\n{round(train_class_distribution, PRECISION)}')
    print()
    print(f'Distribution of classes (Validation):\n{round(valid_class_distribution, PRECISION)}')
    print()


    # Compute sparsity of data
    train_sparsity = (x_train == 0).mean()
    valid_sparsity = (x_valid == 0).mean()
    print(f'Sparsity (Training): {round(train_sparsity * 100, PRECISION)}%')
    print(f'Sparsity (Validation): {round(valid_sparsity * 100, PRECISION)}%')
    print()


    # Image resolution
    res_x, res_y = x_train[0].shape
    print(f'Image resolution: ({res_x}, {res_y}) pixels')
    print()


    # Plot random training images in a grid
    if plot:
        n_rows, n_cols = 5, 5
        n_images = n_rows * n_cols

        indices = list(range(n_train))
        np.random.shuffle(indices)
        indices = indices[:n_images]

        print(f'Plot {n_images} training images...')
        plot_image_grid(x_train[indices], y_train[indices], n_rows, n_cols)
        print()






# ************************************************** HYPERPARAMETER SEARCH ************************************************** #
def compute_new_grid(prev_results, grids):
    new_grid = []

    if prev_results is not None:
        print('Computing models to train...')
        for parameters in ParameterGrid(grids):
            entry = None

            for param_name, param_value in parameters.items():
                df = entry if entry is not None else prev_results
                param_df = df[f'param_{param_name}'].fillna(-1)
                
                if param_value is not None:
                    param_df = param_df.astype(type(param_value))
                    entry = df[param_df == param_value]
                else:
                    entry = df[param_df == -1]

            if len(entry) == 0:
                new_grid += [parameters]

    # No previous results
    else:
        new_grid = list(ParameterGrid(grids))

    # Each parameter value has to be within a list
    for parameters in new_grid:
        for param_name, param_value in parameters.items():
            parameters[param_name] = [param_value]

    return new_grid



def tune_hyperparameters(name, model, grids, data, filename):
    print(f'Tuning hyperparameters for: {name}')

    # Extract data based on split
    [x_train, y_train] = [data['Train'][i] for i in ['x', 'y']]
    [x_valid, y_valid] = [data['Valid'][i] for i in ['x', 'y']]

    print(x_train.shape)

    # Merge training and validation data together
    x = np.concatenate((x_train, x_valid), axis = AXIS_ROW)
    y = np.concatenate((y_train, y_valid), axis = AXIS_ROW)

    # Compute predefined split for grid search (training + validation)
    [n_train, n_valid] = [len(y) for y in [y_train, y_valid]]

    valid_folds = np.zeros(n_train + n_valid)
    valid_folds[:n_train] = -1

    ps = PredefinedSplit(valid_folds)


    # Load previously computed parameters
    path = f'{RESULTS_PATH}/{filename}.csv'
    prev_results = load_dataframe(path)


    # Remove parameters that were already computed for from the grid search
    new_grid = compute_new_grid(prev_results, grids)


    # New parameters to train model with
    if len(new_grid) > 0:

        # Grid search on parameters
        search = GridSearchCV(model, scoring = SCORING, param_grid = new_grid, cv = ps, return_train_score = True, n_jobs = -1, verbose = 3)
        search.fit(x, y)
        print()

        # Generate dataframe for search results
        results = pd.DataFrame(search.cv_results_)

        # Merge old and new results
        if prev_results is not None:
            results = pd.concat((prev_results, results), ignore_index = True)

        # Store results
        store_dataframe(results, path)

    # No new parameters
    else:
        results = prev_results


    # Result associated with best model
    best_results = results.iloc[results['mean_test_score'].idxmax(), :]
    best_params = best_results['params']
    
    # Convert best hyperparameters string to dict
    if type(best_params) is str:
        best_params = best_params.replace('None', "\'None\'").replace('True', "\'True\'").replace('False', "\'False\'")
        best_params = json.loads(best_params.replace('\'', '\"'))

    for name, value in best_params.items():
        if value == 'True' or value == 'False':
            best_params[name] = bool(value)
        if value == 'None':
            best_params[name] = None

    print(f"Best hyperparameters:\n{best_params}")

    # Re-train model with best hyperparameters found
    model.set_params(**best_params)
    model.fit(x_train, y_train)

    return [results, model]



def evaluate_model(model, data):
    [x_train, y_train] = [data['Train'][i] for i in ['x', 'y']]
    [x_valid, y_valid] = [data['Valid'][i] for i in ['x', 'y']]
    [x_test, y_test] = [data['Test'][i] for i in ['x', 'y']]

    # Re-train model using best hyperparameters
    model.fit(x_train, y_train)

    # Predict labels and compute corresponding F1-scores
    f1_train = compute_f1_score(y_train, model.predict(x_train), average = 'macro')
    f1_valid = compute_f1_score(y_valid, model.predict(x_valid), average = 'macro')
    f1_test = compute_f1_score(y_test, model.predict(x_test), average = 'macro')

    return { 'Train': f1_train, 'Valid': f1_valid, 'Test': f1_test }






# ************************************************** TUNING OF MODELS ************************************************** #
def tune_decision_tree(data):

    # Define model
    name = 'Decision Tree'
    model = DecisionTreeClassifier()

    # Max tree depth
    print(f'Theoretical maximum tree depth for training dataset: {MAX_DECISION_TREE_DEPTH}')
    max_depth = np.linspace(1, 60, 60).astype(int)

    # Grid search parameters
    grids = [{
        'criterion': ['gini'],
        'splitter': ['random'],
        'class_weight': ['balanced'],
        'max_depth': max_depth,
    }]

    # Run grid search for model
    [results, best_model] = tune_hyperparameters(name, model, grids, data, f'DT')

    # Plot performance of various classifiers previously trained
    #plot_classifier_performance(name, bow, results, 'max_depth', {
    #    'criterion': ['gini', 'entropy'],
    #})

    # Evaluate best model on all datasets
    f1s = evaluate_model(best_model, data)
    print_f1s(f1s, name)
    print()



def tune_neural_network(data):
    [x_train, y_train] = [data['Train'][i] for i in ['x', 'y']]

    # Use tower model for deep layers
    # First layer: 9216
    # Last layer: 11
    # Number of hidden layers: 9
    n_hidden_layers = 9
    hidden_layer_sizes = [int(N_PIXELS / 2**(i + 1)) for i in range(n_hidden_layers)]

    # Define model
    name = 'Multi-Layer Perceptron'
    model = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes, activation = 'relu', verbose = 3)
    
    # Train NN
    model.fit(x_train, y_train)

    # Evaluate best model on all datasets
    f1s = evaluate_model(model, data)
    print_f1s(f1s, name)
    print()

    return model






def main():

    # Read data
    data = load_data()
    flat_data = flatten_data(data)


    # Explore dataset
    explore_data(data)


    # Tune models
    tune_decision_tree(flat_data)
    #tune_neural_network(flat_data)






if __name__ == '__main__':
    main()