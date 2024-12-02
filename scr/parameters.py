def parameter_grids():
    params = {
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "Decision Tree": {
            "criterion": ["squared_error", "friedman_mse", "absolute_error"],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5, 10]
        },
        "Linear Regression":{},
        
        "KNeighbors Regressor": {
            "n_neighbors": [3, 5, 10, 20],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"]
        },
        "XGB Regressor": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.5, 0.7, 1.0]
        },
        "CatBoost": {
            "iterations": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "depth": [4, 6, 8],
            "l2_leaf_reg": [1, 3, 5]
        },
        "AdaBoost Regressor": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "loss": ["linear", "square", "exponential"]
        }
    }
    return params
