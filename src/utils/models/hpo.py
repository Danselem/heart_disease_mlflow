"""Modeling Optimization with Hyperopt"""
import numpy as np
import yaml
from catboost import CatBoostClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

seed = 1024

def classification_objective(x_train: ArrayLike, y_train: ArrayLike,
                             model_family: str, loss_function: str,
                             params: dict) -> dict:
    """Trainable function for classification models.

    Args:
        x_train (ArrayLike): Training features
        y_train (ArrayLike): Training target
        model_family (str): Model family to optimize
        loss_function (str): Loss function to optimize
        params: Hyperparameter dictionary for the given model.

    Returns:
        A dictionary containing the metrics from training.
    """
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=seed)

    if model_family == 'catboost':
        # Cast integer params from float to int
        integer_params = ['depth', 'min_data_in_leaf', 'max_bin']
        for param in integer_params:
            if param in params:
                params[param] = int(params[param])

        # Extract nested conditional parameters
        if params['bootstrap_type']['bootstrap_type'] == 'Bayesian':
            bagging_temp = params['bootstrap_type'].get(
                'bagging_temperature')
            params['bagging_temperature'] = bagging_temp

        if params['grow_policy']['grow_policy'] == 'LossGuide':
            max_leaves = params['grow_policy'].get('max_leaves')
            params['max_leaves'] = int(max_leaves)

        params['bootstrap_type'] = params['bootstrap_type'][
            'bootstrap_type']
        params['grow_policy'] = params['grow_policy']['grow_policy']

        # Random_strength cannot be < 0
        params['random_strength'] = max(params['random_strength'], 0)
        # fold_len_multiplier cannot be < 1
        params['fold_len_multiplier'] = max(
            params['fold_len_multiplier'], 1)

        model = CatBoostClassifier(**params, verbose=False)
        # Train the model
        model.fit(x_train, y_train, eval_set=(x_val, y_val))

        # Predict on the validation set
        y_pred = model.predict(x_val)

    elif model_family in ['xgboost', 'random_forest']:
        if model_family == 'xgboost':
            model = XGBClassifier(**params, enable_categorical=True)
        else:
            # Create Random Forest model
            model = RandomForestClassifier(**params,)
        
        
        model.fit(x_train, y_train)

        # Predict on the validation set
        y_pred = model.predict(x_val)

    else:
        raise ValueError(f"Unsupported model_family '{model_family}'. "
                         "Supported families are 'catboost', 'xgboost', and 'random_forest'.")

    # Calculate the loss
    if loss_function == 'F1':
        loss = 1 - f1_score(y_val, y_pred, pos_label=1)
    elif loss_function == 'Accuracy':
        loss = 1 - accuracy_score(y_val, y_pred)
    elif loss_function == 'Precision':
        loss = 1 - precision_score(y_val, y_pred, pos_label=1)

    return {'loss': loss, 'status': STATUS_OK}


def classification_optimization(x_train: ArrayLike, y_train: ArrayLike,
                                model_family: str, loss_function: str,
                                objective_function: str,
                                num_trials: int,
                                diagnostic: bool = False) -> dict:
    """Optimize hyperparameters for a model using Hyperopt."""
    
    if model_family == "random_forest":
        search_space = {
            'max_depth': scope.int(hp.quniform('max_depth', 1, 50, 1)),
            'n_estimators': scope.int(hp.quniform('n_estimators', 10, 100, 1)),
            'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
            'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 5, 1)),
            # 'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
            # 'bootstrap': hp.choice('bootstrap', [True, False]),
            # 'criterion': hp.choice('criterion', ['gini', 'entropy']),
            # 'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample'])
            }

    elif model_family == "catboost":
        bootstrap_type = [
            {'bootstrap_type': 'Bayesian', 'bagging_temperature': hp.loguniform('bagging_temperature', np.log(1), np.log(50))},
            {'bootstrap_type': 'Bernoulli'}
        ]
        leb_list = ['No', 'AnyImprovement']
        grow_policy = [
            {'grow_policy': 'SymmetricTree'},
            {'grow_policy': 'Depthwise'},
            {'grow_policy': 'Lossguide', 'max_leaves': hp.quniform('max_leaves', 2, 64, 1)}
        ]

        search_space = {
            'depth': hp.quniform('depth', 2, 10, 1),
            'max_bin': 254,
            'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0, 100),
            'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 50, 1),
            'random_strength': hp.loguniform('random_strength', np.log(0.005), np.log(15)),
            'bootstrap_type': hp.choice('bootstrap_type', bootstrap_type),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'eval_metric': loss_function,
            'objective': objective_function,
            'leaf_estimation_backtracking': hp.choice('leaf_estimation_backtracking', leb_list),
            'grow_policy': hp.choice('grow_policy', grow_policy),
            'colsample_bylevel': hp.quniform('colsample_bylevel', 0.1, 1, 0.01),
            'fold_len_multiplier': hp.loguniform('fold_len_multiplier', np.log(1.01), np.log(3)),
            'od_type': 'Iter',
            'od_wait': 25,
            'task_type': 'CPU'
        }

    elif model_family == "xgboost":
        search_space = {
            'max_depth': hp.choice("max_depth", np.arange(1, 20, 1, dtype=int)),
            'eta': hp.uniform("eta", 0, 1),
            'gamma': hp.uniform("gamma", 0, 100),  # Adjusted upper bound for clarity
            'reg_alpha': hp.uniform("reg_alpha", 1e-7, 10),
            'reg_lambda': hp.uniform("reg_lambda", 0, 1),
            'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
            'colsample_bynode': hp.uniform("colsample_bynode", 0.5, 1),
            'colsample_bylevel': hp.uniform("colsample_bylevel", 0.5, 1),
            'n_estimators': hp.choice("n_estimators", np.arange(10, 1000, 10, dtype=int)),
            'learning_rate': hp.quniform('learning_rate', 0.001, 0.3, 0.01),  # Will need to convert to float
            'min_child_weight': hp.choice("min_child_weight", np.arange(1, 10, 1, dtype=int)),
            'max_delta_step': hp.choice("max_delta_step", np.arange(0, 10, 1, dtype=int)),  # Adjusted range
            'subsample': hp.uniform("subsample", 0.5, 1),
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'random_state': seed
            }

    else:
        raise ValueError(f"Unsupported model_family '{model_family}'. Supported families are 'random_forest', 'catboost', and 'xgboost'.")

    rstate = np.random.default_rng(seed)
    trials = Trials()
    best_params = fmin(
        fn=lambda params: classification_objective(
            x_train, y_train, model_family, loss_function, params),
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=trials,
        rstate=rstate
    )

    # Convert and handle parameters for all models
    if model_family == "catboost":
        best_params['bootstrap_type'] = bootstrap_type[
            best_params['bootstrap_type']]['bootstrap_type']
        best_params['grow_policy'] = grow_policy[
            best_params['grow_policy']]['grow_policy']
        best_params['eval_metric'] = loss_function
        best_params['leaf_estimation_backtracking'] = leb_list[
            best_params['leaf_estimation_backtracking']]
        integer_params = ['depth', 'min_data_in_leaf', 'max_bin']
        for param in integer_params:
            if param in best_params:
                best_params[param] = int(best_params[param])
        if 'max_leaves' in best_params:
            best_params['max_leaves'] = int(best_params['max_leaves'])
        print('{' + '\n'.join('{}: {}'.format(k, v)
              for k, v in best_params.items()) + '}')

        if diagnostic:
            for i, trial in enumerate(trials.trials):
                print(f"Trial # {i} result: {trial['result']['loss']}")
                
    elif model_family == "random_forest":
        
        # Convert specified parameters to integers if they are present
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])



        if diagnostic:
            for i, trial in enumerate(trials.trials):
                print(f"Trial # {i} result: {trial['result']['loss']}")

    elif model_family == "xgboost":
        # Convert and handle parameters for xgboost
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['eta'] = float(best_params['eta'])
        best_params['gamma'] = float(best_params['gamma'])
        best_params['reg_alpha'] = float(best_params['reg_alpha'])
        best_params['reg_lambda'] = float(best_params['reg_lambda'])
        best_params['colsample_bytree'] = float(best_params['colsample_bytree'])
        best_params['colsample_bynode'] = float(best_params['colsample_bynode'])
        best_params['colsample_bylevel'] = float(best_params['colsample_bylevel'])
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['learning_rate'] = float(best_params['learning_rate'])
        best_params['min_child_weight'] = int(best_params['min_child_weight'])
        best_params['max_delta_step'] = int(best_params['max_delta_step'])
        best_params['subsample'] = float(best_params['subsample'])
        best_params['objective'] = best_params.get('objective', 'binary:logistic')
        best_params['eval_metric'] = best_params.get('eval_metric', 'aucpr')

        if diagnostic:
            for i, trial in enumerate(trials.trials):
                print(f"Trial # {i} result: {trial['result']['loss']}")


    return best_params
