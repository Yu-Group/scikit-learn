#!/usr/bin/python
from sklearn.tree import irf_utils
import numpy as np
from functools import reduce
# For the generate_rf_example function
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_true


def generate_rf_example(sklearn_ds=load_breast_cancer(),
                        train_split_propn=0.9,
                        n_estimators=3,
                        feature_weight=None,
                        random_state_split=2017,
                        random_state_classifier=2018):
    """
    This fits a random forest classifier to the breast cancer/ iris datasets
    This can be called from the jupyter notebook so that analysis
    can take place quickly
    Parameters
    ----------
    sklearn_ds : sklearn dataset
        Choose from the `load_breast_cancer` or the `load_iris datasets`
        functions from the `sklearn.datasets` module
    train_split_propn : float
        Should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split.
    n_estimators : int, optional (default=10)
        The index of the root node of the tree. Should be set as default to
        3 and not changed by the user
    feature_weight : list, optional (default=None)
        The chance of splitting at each feature.
    random_state_split: int (default=2017)
        The seed used by the random number generator for the `train_test_split`
        function in creating our training and validation sets
    random_state_classifier: int (default=2018)
        The seed used by the random number generator for
        the `RandomForestClassifier` function in fitting the random forest
    Returns
    -------
    X_train : array-like or sparse matrix, shape = [n_samples, n_features]
        Training features vector, where n_samples in the number of samples and
        n_features is the number of features.
    X_test : array-like or sparse matrix, shape = [n_samples, n_features]
        Test (validation) features vector, where n_samples in the
        number of samples and n_features is the number of features.
    y_train : array-like or sparse matrix, shape = [n_samples, n_classes]
        Training labels vector, where n_samples in the number of samples and
        n_classes is the number of classes.
    y_test : array-like or sparse matrix, shape = [n_samples, n_classes]
        Test (validation) labels vector, where n_samples in the
        number of samples and n_classes is the number of classes.
    rf : RandomForestClassifier object
        The fitted random forest to the training data
    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> X_train, X_test, y_train, y_test,
        rf = generate_rf_example(sklearn_ds =
                                load_breast_cancer())
    >>> print(X_train.shape)
    ...                             # doctest: +SKIP
    ...
    (512, 30)
    """

    # Load the relevant scikit learn data
    raw_data = sklearn_ds

    # Create the train-test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        raw_data.data, raw_data.target, train_size=train_split_propn,
        random_state=random_state_split)

    # Just fit a simple random forest classifier with 2 decision trees
    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state_classifier)

    # fit the classifier
    if feature_weight is None:
        rf.fit(X=X_train, y=y_train)
    else:
        rf.fit(X=X_train, y=y_train, feature_weight=feature_weight)

    return X_train, X_test, y_train, y_test, rf


# Load the breast cancer dataset
breast_cancer = load_breast_cancer()

# Generate the training and test datasets
X_train, X_test, y_train, \
    y_test, rf = generate_rf_example(
        sklearn_ds=breast_cancer, n_estimators=10)

# Get all of the random forest and decision tree data
all_rf_tree_data = irf_utils.get_rf_tree_data(rf=rf,
                                              X_train=X_train,
                                              X_test=X_test,
                                              y_test=y_test)

# Get the RIT data and produce RITs
np.random.seed(12)
gen_random_leaf_paths = irf_utils.generate_rit_samples(
    all_rf_tree_data=all_rf_tree_data,
    bin_class_type=1)

# Build single Random Intersection Tree
# This is not using noisy splits i.e. 5 splits per node
rit0 = irf_utils.build_tree(
    feature_paths=gen_random_leaf_paths,
    max_depth=3,
    noisy_split=False,
    num_splits=5)

# Build single Random Intersection T
# This is using noisy splits i.e. {5, 6} splits per node
rit1 = irf_utils.build_tree(
    max_depth=3,
    noisy_split=True,
    feature_paths=gen_random_leaf_paths,
    num_splits=5)

# Build single Random Intersection Tree of depth 1
# This is not using noisy splits i.e. 5 splits per node
# This should only have a single (root) node
rit2 = irf_utils.build_tree(
    max_depth=1,
    noisy_split=True,
    feature_paths=gen_random_leaf_paths,
    num_splits=5)

# Get the entire RIT data
np.random.seed(12)
all_rit_tree_data = irf_utils.get_rit_tree_data(
    all_rf_tree_data=all_rf_tree_data,
    bin_class_type=1,
    M=10,
    max_depth=3,
    noisy_split=False,
    num_splits=2)

# Manually construct an RIT example
# Get the unique feature paths where the leaf
# node predicted class is just 1
# We are just going to get it from the first decision tree
# for this test case
uniq_feature_paths \
    = all_rf_tree_data['dtree0']['all_uniq_leaf_paths_features']
leaf_node_classes \
    = all_rf_tree_data['dtree0']['all_leaf_node_classes']
ones_only \
    = [i for i, j in zip(uniq_feature_paths, leaf_node_classes)
       if j == 1]

# Manually extract the last seven values for our example
# Just pick the last seven cases
# we are going to manually construct
# We are going to build a BINARY RIT of depth 3
# i.e. max `2**3 -1 = 7` intersecting nodes
ones_only_seven = ones_only[-7:]

# Manually build the RIT
# Construct a binary version of the RIT manually!
node0 = ones_only_seven[0]
node1 = np.intersect1d(node0, ones_only_seven[1])
node2 = np.intersect1d(node1, ones_only_seven[2])
node3 = np.intersect1d(node1, ones_only_seven[3])
node4 = np.intersect1d(node0, ones_only_seven[4])
node5 = np.intersect1d(node4, ones_only_seven[5])
node6 = np.intersect1d(node4, ones_only_seven[6])

intersected_nodes_seven \
    = [node0, node1, node2, node3, node4, node5, node6]

leaf_nodes_seven = [node2, node3, node5, node6]

rit_output \
    = reduce(np.union1d, (node2, node3, node5, node6))

# Now we can create the RIT using our built irf_utils
# build the generator of 7 values
ones_only_seven_gen = (n for n in ones_only_seven)

# Build the binary RIT using our irf_utils
rit_man0 = irf_utils.build_tree(
    feature_paths=ones_only_seven_gen,
    max_depth=3,
    noisy_split=False,
    num_splits=2)

# Calculate the union values

# First on the manually constructed RIT
rit_union_output_manual \
    = reduce(np.union1d, (node2, node3, node5, node6))

# Lastly on the RIT constructed using a function
rit_man0_union_output \
    = reduce(np.union1d, [node[1]._val
                          for node in rit_man0.leaf_nodes()])

def test_manual_binary_RIT1():
    # Test the manually constructed binary RIT
    # Check all node values
    assert_equal([node[1]._val.tolist()
                  for node in rit_man0.traverse_depth_first()],
                 [node.tolist()
                  for node in intersected_nodes_seven])

    # Check all leaf node intersected values
    assert_equal([node[1]._val.tolist()
                  for node in rit_man0.leaf_nodes()],
                 [node.tolist() for node in leaf_nodes_seven])

    # Check the union value calculation
    assert_equal(rit_union_output_manual.tolist(),
                 rit_man0_union_output.tolist())



# Test that the train test observations sum to the
# total data set observations


def test_generate_rf_example1():

    # Check train test feature split from `generate_rf_example`
    # against the original breast cancer dataset
    assert_equal(X_train.shape[0] + X_test.shape[0],
                 breast_cancer.data.shape[0])

    assert_equal(X_train.shape[1],
                 breast_cancer.data.shape[1])

    assert_equal(X_test.shape[1],
                 breast_cancer.data.shape[1])

    # Check feature and outcome sizes
    assert_equal(X_train.shape[0] + X_test.shape[0],
                 y_train.shape[0] + y_test.shape[0])



# Test build RIT
def test_build_tree():
    assert_true(len(rit0) <= 1 + 5 + 5**2)
    assert_true(len(rit1) <= 1 + 6 + 6**2)
    assert_true(len(rit2) == 1)


def test_rf_output():
    leaf_node_path = [[0, 1, 2, 3, 4, 5],
                      [0, 1, 2, 3, 4, 6, 7, 8],
                      [0, 1, 2, 3, 4, 6, 7, 9, 10, 11],
                      [0, 1, 2, 3, 4, 6, 7, 9, 10, 12],
                      [0, 1, 2, 3, 4, 6, 7, 9, 13],
                      [0, 1, 2, 3, 4, 6, 14, 15],
                      [0, 1, 2, 3, 4, 6, 14, 16],
                      [0, 1, 2, 3, 17, 18],
                      [0, 1, 2, 3, 17, 19],
                      [0, 1, 2, 20, 21, 22, 23],
                      [0, 1, 2, 20, 21, 22, 24, 25],
                      [0, 1, 2, 20, 21, 22, 24, 26],
                      [0, 1, 2, 20, 21, 27],
                      [0, 1, 2, 20, 28],
                      [0, 1, 29, 30],
                      [0, 1, 29, 31],
                      [0, 32, 33, 34, 35],
                      [0, 32, 33, 34, 36],
                      [0, 32, 33, 37, 38],
                      [0, 32, 33, 37, 39],
                      [0, 32, 40]]

    leaf_node_samples = [114, 1, 3, 1, 67, 1, 1,
                         1, 3, 2, 3, 1, 3, 7, 2, 7, 1, 11, 3, 1, 91]

    leaf_node_values = [[0, 189],
                        [3, 0],
                        [0, 5],
                        [1, 0],
                        [0, 101],
                        [1, 0],
                        [0, 1],
                        [2, 0],
                        [0, 3],
                        [0, 2],
                        [5, 0],
                        [0, 1], [0, 7], [10, 0], [0, 3],
                        [12, 0], [0, 2],
                        [19, 0], [0, 7], [1, 0], [137, 0]]

    leaf_paths_features = [[20, 24, 27, 10, 0],
                           [20, 24, 27, 10, 0, 6, 0],
                           [20, 24, 27, 10, 0, 6, 0, 14, 20],
                           [20, 24, 27, 10, 0, 6, 0, 14, 20],
                           [20, 24, 27, 10, 0, 6, 0, 14],
                           [20, 24, 27, 10, 0, 6, 18],
                           [20, 24, 27, 10, 0, 6, 18],
                           [20, 24, 27, 10, 28],
                           [20, 24, 27, 10, 28],
                           [20, 24, 27, 21, 6, 6],
                           [20, 24, 27, 21, 6, 6, 12],
                           [20, 24, 27, 21, 6, 6, 12],
                           [20, 24, 27, 21, 6],
                           [20, 24, 27, 21],
                           [20, 24, 22], [20, 24, 22],
                           [20, 7, 17, 29],
                           [20, 7, 17, 29],
                           [20, 7, 17, 28],
                           [20, 7, 17, 28],
                           [20, 7]]

    node_depths = [5, 7, 9, 9, 8, 7, 7, 5, 5,
                   6, 7, 7, 5, 4, 3, 3, 4, 4, 4, 4, 2]

    assert_array_equal(   np.concatenate(all_rf_tree_data['dtree1']['all_leaf_node_paths']),
                          np.concatenate(leaf_node_path))

    assert_array_equal(all_rf_tree_data['dtree1']['all_leaf_node_samples'],
                       leaf_node_samples)

    assert_array_equal(np.concatenate(
        all_rf_tree_data['dtree1']['all_leaf_node_values'], axis=0),
                       leaf_node_values)

    assert_array_equal(np.concatenate(
        all_rf_tree_data['dtree1']['all_leaf_paths_features']),
                       np.concatenate(leaf_paths_features))

    assert_array_equal(node_depths,
                       all_rf_tree_data['dtree1']['leaf_nodes_depths'])


# test RIT_interactions
def test_rit_interactions():
    all_rit_tree_data_test = {'rit0':
                              {'rit_intersected_values':
                               [np.array([1, 2, 3]),
                                np.array([1, 2])]},
                              'rit1':
                              {'rit_intersected_values':
                               [np.array([1, 2, 3, 4]),
                                np.array([1, 2, 3])]},
                              'rit2':
                              {'rit_intersected_values':
                               [np.array([1, 2]),
                                np.array([5, 6]), np.array([])]},
                              'rit3':
                              {'rit_intersected_values':
                               [np.array([1, 2, 3]),
                                np.array([1, 2, 3, 4])]},
                              'rit4':
                              {'rit_intersected_values':
                               [np.array([1, 2, 3]),
                                np.array([1, 2, 3])]}}

    output = irf_utils.rit_interactions(all_rit_tree_data_test)

    L1 = output.keys()

    L3 = ['1_2_3', '1_2', '1_2_3_4', '5_6']
    L4 = [5, 2, 2, 1]
    output_test = dict(zip(L3, L4))

    # check keys
    assert_true(len(L1) == len(L3) and sorted(L1) == sorted(L3))

    # check values
    for key in output.keys():
        assert_true(output[key] == output_test[key])
