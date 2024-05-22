import numpy as np
import pandas as pd 
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.metrics import mean_squared_log_error
import geopy.distance
import time



def describe_column(meta):
    """
    Utility function for describing a dataset column (see below for usage)
    """
    def f(x):
        d = pd.Series(name=x.name, dtype=object)
        m = next(m for m in meta if m['name'] == x.name)
        d['Type'] = m['type']
        d['#NaN'] = x.isna().sum()
        d['Description'] = m['desc']
        if m['type'] == 'categorical':
            counts = x.dropna().map(dict(enumerate(m['cats']))).value_counts().sort_index()
            d['Statistics'] = ', '.join(f'{c}({n})' for c, n in counts.items())
        elif m['type'] == 'real' or m['type'] == 'integer':
            stats = x.dropna().agg(['mean', 'std', 'min', 'max'])
            d['Statistics'] = ', '.join(f'{s}={v :.1f}' for s, v in stats.items())
        elif m['type'] == 'boolean':
            counts = x.dropna().astype(bool).value_counts().sort_index()
            d['Statistics'] = ', '.join(f'{c}({n})' for c, n in counts.items())
        else:
            d['Statistics'] = f'#unique={x.nunique()}'
        return d
    return f


def describe_data(data, meta):
    desc = data.apply(describe_column(meta)).T
    desc = desc.style.set_properties(**{'text-align': 'left'}) # set CSS properties for each table cell
    desc = desc.set_table_styles([ dict(selector='th', props=[('text-align', 'left')])]) # set table style
    return desc


def rmsle(y_true, y_pred, logged):
    """
    Calculate root mean squared log error.
    Alternativerly: sklearn.metrics.mean_squared_log_error(y_true, y_pred) ** 0.5
    """
    assert (y_true >= 0).all()
    assert (y_pred >= 0).all()
    if logged:
        log_error = y_pred - y_true
    else:
        log_error = np.log1p(y_pred) - np.log1p(y_true)  # Note: log1p(x) = log(1 + x)
    return np.mean(log_error ** 2) ** 0.5


def rmsle_flaml(X_test, y_test, estimator, labels, X_train, y_train,
                weight_test=None, weight_train=None, config=None,
                groups_test=None, groups_train=None):
    """
    Custom metric function for FLAML's AutoML framework
    """
    from sklearn.metrics import mean_squared_log_error    
    y_pred = estimator.predict(X_test)
    
    if (y_pred < 0).any():
        y_pred[y_pred < 0] = 0
    
    rmsle = mean_squared_log_error(y_test, y_pred) ** 0.5    
    return rmsle, { "rmsle" : rmsle }

def rmsle_flaml_alt(X_test, y_test, estimator, labels, X_train, y_train,
                weight_test=None, weight_train=None, config=None,
                groups_test=None, groups_train=None):
    """
    Alternative custom metric function for FLAML's AutoML framework
    """  
    y_pred = estimator.predict(X_test)
    
    if (y_pred < 0).any():
        y_pred[y_pred < 0] = 0
    
    log_error = y_pred - y_test
    rmsle = np.mean(log_error ** 2) ** 0.5   
    return rmsle, { "rmsle" : rmsle }

def rmsle_lgbm(y_true, y_pred):
    """
    Custom evaluation metric function for LightGBM
    """    
    rmsle = mean_squared_log_error(y_true, y_pred) ** 0.5
    return "RMSLE", rmsle, False


def cross_validated_rmsle(clf, x_train, y_train, n_folds=5, seed=123, outlier_limit=20):
    """
    Calculate the mean RMSLE given a training set.
    
    Trains a classifier on 4/5 of the training data and
    predicts the rest (1/5). This procedure is repeated for all 5 folds,
    thus we have predictions for all training set. This prediction is one
    column of meta-data, later on used as a feature column by a meta-algorithm.
    We predict the test part and average predictions across all 5 models.
    
    Keyword arguments:
    clf -- classifier
    x_train -- training data without labels
    y_train -- labels for the training data
    n_folds -- number of folds for cross validation
    seed -- random seed, for reproducibility

    Returns:
    Average RMSLE across all folds.
    
    """
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed
    )

    ntrain = x_train.shape[0]

    x_train = x_train.values
    y_train = y_train.ravel()

    rmsle_scores = np.empty((n_folds,))

    y_strat = y_train.round()
    # bundle all high outliers in one class
    y_strat[y_strat > outlier_limit] = outlier_limit + 1

    for i, (train_index, test_index) in enumerate(skf.split(x_train, y_strat)):
        x_train_fold = x_train[train_index]
        y_train_fold = y_train[train_index]
        x_test_fold = x_train[test_index]
        y_test_fold = y_train[test_index]

        clf.fit(x_train_fold, y_train_fold)

        y_fold_hat = clf.predict(x_test_fold) # predict on the test set

        # replace negative predictions with zeroes
        if (y_fold_hat < 0).any():
            print("WARNING: Negative prediction(s) given!")
            print(y_fold_hat[y_fold_hat < 0])
            y_fold_hat[y_fold_hat < 0] = 0

        rmsle_scores[i] = rmsle(y_true=y_test_fold, y_pred=y_fold_hat, logged=True)

    return rmsle_scores.mean(axis=0)


def cross_validated_lgbm(clf, x_train, y_train, cat_idx, n_folds=5, seed=123, outlier_limit=20):
    
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed
    )

    ntrain = x_train.shape[0]

    x_train = x_train.values
    y_train = y_train.ravel()

    rmsle_scores = np.empty((n_folds,))

    y_strat = y_train.round()
    # bundle all high outliers in one class
    y_strat[y_strat > outlier_limit] = outlier_limit + 1
    for i, (train_index, test_index) in enumerate(skf.split(x_train, y_strat)):
        print("=" * 12 + f"Training fold {i}" + 12 * "=")
        start = time.time()

        x_train_fold = x_train[train_index]
        y_train_fold = y_train[train_index]
        x_test_fold = x_train[test_index]
        y_test_fold = y_train[test_index]

        clf.fit(x_train_fold, y_train_fold, categorical_feature=cat_idx)

        y_fold_hat = clf.predict(x_test_fold) # predict on the test set

        # replace negative predictions with zeroes
        if (y_fold_hat < 0).any():
            print("WARNING: Negative prediction(s) given!")
            print(y_fold_hat[y_fold_hat < 0])
            y_fold_hat[y_fold_hat < 0] = 0

        rmsle_scores[i] = rmsle(y_true=y_test_fold, y_pred=y_fold_hat, logged=True)
        runtime = time.time() - start
        print(f"Fold {i} finished with score: {rmsle_scores[i]:.5f} in {runtime:.2f} seconds.\n")
    
    return rmsle_scores.mean(axis=0)


def dist(apartment, center=(55.75393226, 37.62079233)):
    """
    Get the distance between 2 sets of coordinates.
    
    Params:
    apartment -- tuple of the apartment coordinates in format
        (latitude, longitude)
    center -- another tuple of the coordinates (Red Square by default)

    Returns:
    Distance between two coordinate points in kilometres
    
    """
    return round(geopy.distance.distance(apartment, center).km, 2)


def groupkfold_lgbm(clf, x_train, y_train, cat_idx, groups, n_folds=5, seed=123, outlier_limit=20):
    
    gkf = GroupKFold(n_splits = n_folds)

    ntrain = x_train.shape[0]

    x_train = x_train.values
    y_train = y_train.ravel()

    rmsle_scores = np.empty((n_folds,))

    y_strat = y_train.round()
    # bundle all high outliers in one class
    y_strat[y_strat > outlier_limit] = outlier_limit + 1
    for i, (train_index, test_index) in enumerate(gkf.split(x_train, y_strat, groups=groups)):
        print("=" * 12 + f"Training fold {i}" + 12 * "=")
        start = time.time()

        x_train_fold = x_train[train_index]
        y_train_fold = y_train[train_index]
        x_test_fold = x_train[test_index]
        y_test_fold = y_train[test_index]

        clf.fit(x_train_fold, y_train_fold, categorical_feature=cat_idx)

        y_fold_hat = clf.predict(x_test_fold) # predict on the test set

        # replace negative predictions with zeroes
        if (y_fold_hat < 0).any():
            print("WARNING: Negative prediction(s) given!")
            print(y_fold_hat[y_fold_hat < 0])
            y_fold_hat[y_fold_hat < 0] = 0

        rmsle_scores[i] = rmsle(y_true=y_test_fold, y_pred=y_fold_hat, logged=True)
        runtime = time.time() - start
        print(f"Fold {i} finished with score: {rmsle_scores[i]:.5f} in {runtime:.2f} seconds.\n")
    
    return rmsle_scores.mean(axis=0)


def strat_gkf_lgbm(clf, x_train, y_train, cat_idx, groups, n_folds=5, seed=123, outlier_limit=20):
    
    sgkf = StratifiedGroupKFold(n_splits = n_folds, shuffle=True, random_state=seed)

    ntrain = x_train.shape[0]

    x_train = x_train.values
    y_train = y_train.ravel()

    rmsle_scores = np.empty((n_folds,))

    y_strat = y_train.round()
    # bundle all high outliers in one class
    y_strat[y_strat > outlier_limit] = outlier_limit + 1
    for i, (train_index, test_index) in enumerate(sgkf.split(x_train, y_strat, groups=groups)):
        print("=" * 12 + f"Training fold {i}" + 12 * "=")
        start = time.time()

        x_train_fold = x_train[train_index]
        y_train_fold = y_train[train_index]
        x_test_fold = x_train[test_index]
        y_test_fold = y_train[test_index]

        clf.fit(x_train_fold, y_train_fold, categorical_feature=cat_idx)

        y_fold_hat = clf.predict(x_test_fold) # predict on the test set

        # replace negative predictions with zeroes
        if (y_fold_hat < 0).any():
            print("WARNING: Negative prediction(s) given!")
            print(y_fold_hat[y_fold_hat < 0])
            y_fold_hat[y_fold_hat < 0] = 0

        rmsle_scores[i] = rmsle(y_true=y_test_fold, y_pred=y_fold_hat, logged=True)
        runtime = time.time() - start
        print(f"Fold {i} finished with score: {rmsle_scores[i]:.5f} in {runtime:.2f} seconds.\n")
    
    return rmsle_scores.mean(axis=0)

def rmsle_lgbm2(y_true, y_pred):
    """
    Custom evaluation metric function for LightGBM
    """    
    log_error = y_pred - y_true
    rmsle = np.mean(log_error ** 2) ** 0.5
    return "RMSLE", rmsle, False

def strat_gkf_lgbm_es(clf, x_train, y_train, cat_idx, groups, n_folds=5, seed=123, outlier_limit=20):
    
    sgkf = StratifiedGroupKFold(n_splits = n_folds, shuffle=True, random_state=seed)

    x_train = x_train.values
    y_train = y_train.ravel()

    rmsle_scores = np.empty((n_folds,))

    y_strat = y_train.round()
    # bundle all high outliers in one class
    y_strat[y_strat > outlier_limit] = outlier_limit + 1
    for i, (train_index, test_index) in enumerate(sgkf.split(x_train, y_strat, groups=groups)):
        print("=" * 12 + f"Training fold {i}" + 12 * "=")
        start = time.time()

        x_train_fold = x_train[train_index]
        y_train_fold = y_train[train_index]
        x_test_fold = x_train[test_index]
        y_test_fold = y_train[test_index]

        eval_set = [(x_test_fold, y_test_fold)]

        clf.fit(x_train_fold, y_train_fold,
            categorical_feature=cat_idx,
            eval_set=eval_set,
            early_stopping_rounds=500,
            eval_metric=rmsle_lgbm2
            )

        y_fold_hat = clf.predict(x_test_fold) # predict on the test set

        # replace negative predictions with zeroes
        if (y_fold_hat < 0).any():
            print("WARNING: Negative prediction(s) given!")
            print(y_fold_hat[y_fold_hat < 0])
            y_fold_hat[y_fold_hat < 0] = 0

        rmsle_scores[i] = rmsle(y_true=y_test_fold, y_pred=y_fold_hat, logged=True)
        runtime = time.time() - start
        print(f"Fold {i} finished with score: {rmsle_scores[i]:.5f} in {runtime:.2f} seconds.\n")
    
    return rmsle_scores.mean(axis=0)


def dist_dist(apartment):
    """
    Get the distance between 2 sets of coordinates.

    Params:
    apartment -- tuple of the apartment coordinates and district in format
        (latitude, longitude, district)
    center -- another tuple of the coordinates (Red Square by default)
    Returns:
    Distance between two coordinate points in kilometres

    """
    district_centres = [
        (55.75383701125541, 37.59760507907648), (55.82950216205188, 37.535673869717286),
        (55.852733421924576, 37.62390436552696), (55.78029181151668, 37.7854613533669),
        (55.70392116690081, 37.79598576678843), (55.64602690480349, 37.658203463173216),
        (55.63260370970513, 37.54687435642714), (55.711195179419526, 37.46543980093814),
        (55.80900383756346, 37.44976383692894), (55.974729127320956, 37.17257165251989),
        (55.44408945670997, 37.13442126839827), (55.5678043050145, 37.42770446125155),
        (34.371277666666664, 18.650676311111106)
    ]
    return round(geopy.distance.distance(apartment[0:2], district_centres[int(apartment[2])]).km, 2)


def dist_loc_1(apartment):
    """
    Get the distance between apartment and Sjeremetjevo international airport.

    Params:
    apartment -- tuple of the apartment coordinates in format
        (latitude, longitude)
    Returns:
    Distance between two coordinate points in kilometres

    """
    return round(geopy.distance.distance(apartment, (55.9678, 37.3818)).km, 2)


def dist_loc_2(apartment):
    """
    Get the distance between apartment and Izmaylovo park.

    Params:
    apartment -- tuple of the apartment coordinates in format
        (latitude, longitude)
    Returns:
    Distance between two coordinate points in kilometres

    """
    return round(geopy.distance.distance(apartment, (55.7768, 37.7938)).km, 2)


def dist_loc_3(apartment):
    """
    Get the distance between apartment and Church Khram pok..

    Params:
    apartment -- tuple of the apartment coordinates in format
        (latitude, longitude)
    Returns:
    Distance between two coordinate points in kilometres

    """
    return round(geopy.distance.distance(apartment, (55.7055, 37.9281)).km, 2)


def dist_loc_4(apartment):
    """
    Get the distance between apartment and commercial district..

    Params:
    apartment -- tuple of the apartment coordinates in format
        (latitude, longitude)
    Returns:
    Distance between two coordinate points in kilometres

    """
    return round(geopy.distance.distance(apartment, (55.5478, 37.5529)).km, 2)


def direction(apartment, center=(55.75393226, 37.62079233)):
    """
    Get the direction from city centre. 0 is north, 90 is east, 180 is south and 270 is west.

    :param apartment:
    :param center:
    :return:
    """
    x = np.array([-10, 0])
    y = np.array([apartment[0] - center[0], apartment[1] - center[1]])
    dot = np.dot(x, y)
    det = x[0] * y[1] - x[1] * y[0]
    angle = np.arctan2(det, dot)
    return 180 * angle / np.pi + 180


def delete_suspicious_data(data):
    # Look for indices of weird data sequences in area_living that seem uncredible:

    # mhu.scatter_plot(data, 'area_total', 'area_kitchen', 'area_living', 'delete_0')

    error_indices = find_repeating_area_pattern(data)
    data.at[error_indices, 'area_kitchen'] = np.nan
    data.at[error_indices, 'area_living'] = np.nan

    # mhu.scatter_plot(data, 'area_total', 'area_kitchen', 'area_living', 'delete_1')

    error_indices = get_overly_quantized_indices(data)
    data.at[error_indices, 'area_living'] = np.nan

    # mhu.scatter_plot(data, 'area_total', 'area_kitchen', 'area_living', 'delete_2')

    return data


def find_repeating_area_pattern(data):

    # Find indices with the repeating patterns
    index_set = set()
    for i in range(1, len(data)):
        if data.at[i - 1, 'area_kitchen'] == data.at[i, 'area_kitchen'] == data.at[i + 1, 'area_kitchen']:
            if data.at[i - 1, 'area_living'] == data.at[i, 'area_living'] == data.at[i + 1, 'area_living']:
                if not (data.at[i - 1, 'area_total'] == data.at[i, 'area_total'] == data.at[i + 1, 'area_total']):
                    index_set.add(i - 1)
                    index_set.add(i)
                    index_set.add(i + 1)

    index_list = list(index_set)
    index_list.sort()

    # Exclude clusters smaller than 4, since the repeating pattern can also occur randomly
    number_of_clusters = 0
    clusters = []
    prev_ind = -2
    for index in index_list:
        if index == prev_ind + 1:
            clusters[-1]['indices'].append(index)
            clusters[-1]['size'] += 1
        else:
            number_of_clusters += 1
            clusters.append({'indices': [index], 'size': 1})
        prev_ind = index
    avg_cluster_size = 0
    for cluster in clusters:
        if cluster['size'] <= 4:
            number_of_clusters -= 1
            for index in cluster['indices']:
                index_list.remove(index)
        else:
            avg_cluster_size += cluster['size']
    if number_of_clusters == 0:
        avg_cluster_size = 0
    else:
        avg_cluster_size = avg_cluster_size / number_of_clusters

    return index_list


def get_overly_quantized_indices(data):

    index_set = set()
    # for i in range(1, len(data)):
    #     if data.at[i, 'area_kitchen'] in [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0]:
    #         index_set.add(i)

    for i in range(1, len(data)):
        if data.at[i, 'area_living'] == 0.0:
            index_set.add(i)

    index_list = list(index_set)
    index_list.sort()

    return index_list
  
  
