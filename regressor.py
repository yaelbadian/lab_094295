from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import pickle
from time import time
from numpy.random import uniform
import seaborn as sns
sns.set()

class Regressor:
    def __init__(self, model='xgboost', init_points=3, n_iter=2):
        if not isinstance(model, str):
            raise Exception("model should be a string")
        self._model_str = model.lower().replace(' ', '')
        if self._model_str in ('bestregressor', 'regressor'):
            self._init_function = None
            self._optimize_function = None
            self._model_str = 'regressor'
        elif self._model_str in ('xgboost', 'xgbreg', 'xgboostregressor', 'xgbregressor'):
            self._init_function = self._init_XGBRegressor
            self._optimize_function = self._optimize_XGBRegressor
            self._model_str = 'xgbreg'
        elif self._model_str in ('rfr', 'randomforestregressor'):
            self._init_function = self._init_RandomForestRegressor
            self._optimize_function = self._optimize_RandomForestRegressor
            self._model_str = 'rfr'
        elif self._model_str in ('linearregression', 'lasso'):
            self._init_function = self._init_Lasso
            self._optimize_function = self._optimize_Lasso
            self._model_str = 'lasso'
        else:
            raise Exception("Model should be one of the following:'xgboost', 'rfr', 'lr', 'lasso'")
        self._fitted_model = None
        self._feature_importances_ = None
        self._init_points = init_points
        self._n_iter = n_iter

    def _bayesian_optimization(self, cv_function, parameters):
        gp_params = {"alpha": 1e-5, 'init_points': self._init_points, 'n_iter': self._n_iter}
        bo = BayesianOptimization(cv_function, parameters)
        bo.maximize(**gp_params)
        return bo.max


    # -------------- init & optimize functions -------------- #

    # ----- rfr -----

    @staticmethod
    def _init_RandomForestRegressor(params):
        return RandomForestRegressor(
            n_estimators=int(max(params['n_estimators'], 1)),
            max_depth=int(max(params['max_depth'], 1)),
            min_samples_split=int(max(params['min_samples_split'], 2)),
            min_samples_leaf=int(max(params['min_samples_leaf'], 2)),
            n_jobs=-1,
            random_state=42)

    @staticmethod
    def _optimize_RandomForestRegressor(X, y, cv_splits=5, scoring='neg_mean_squared_error', n_jobs=-1):
        def cv_function(n_estimators, max_depth, min_samples_split, min_samples_leaf):
            params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf}
            return cross_val_score(Regressor._init_RandomForestRegressor(params), X=X, y=y, cv=cv_splits,
                                   scoring=scoring, n_jobs=n_jobs).mean()

        parameters = {"n_estimators": (10, 100),
                      "max_depth": (5, 100),
                      "min_samples_split": (2, 100),
                      "min_samples_leaf": (2, 50)}
        return cv_function, parameters

    # ----- xgboost regressor ----- #

    @staticmethod
    def _init_XGBRegressor(params, objective='reg:squarederror'):
        return XGBRegressor(
            objective=objective,
            learning_rate=max(params['eta'], 0),
            gamma=max(params['gamma'], 0),
            max_depth=int(max(params['max_depth'], 1)),
            n_estimators=int(max(params['n_estimators'], 1)),
            min_child_weight=int(max(params['min_child_weight'], 1)),
            seed=42,
            nthread=-1)

    @staticmethod
    def _optimize_XGBRegressor(X, y, cv_splits=5, scoring='neg_mean_squared_error', n_jobs=-1):
        def cv_function(eta, gamma, max_depth, n_estimators, min_child_weight):
            params = {'eta': eta, 'gamma': gamma, 'max_depth': max_depth, 'n_estimators': n_estimators,
                      'min_child_weight': min_child_weight}
            return cross_val_score(Regressor._init_XGBRegressor(params), X=X, y=y, cv=cv_splits,
                                   scoring=scoring, n_jobs=n_jobs).mean()

        parameters = {"eta": (0.001, 0.4),
                      "gamma": (0, 15),
                      "max_depth": (1, 75),
                      "n_estimators": (1, 150),
                      "min_child_weight": (1, 20)}
        return cv_function, parameters

    # ----- lasso ----- #

    @staticmethod
    def _init_Lasso(params):
        if params['alpha'] < 0.25:
            return LinearRegression(n_jobs=-1)
        else:
            return Lasso(alpha=max(params['alpha'], 0.25))

    @staticmethod
    def _optimize_Lasso(X, y, cv_splits=5, scoring='neg_mean_squared_error', n_jobs=-1):
        def cv_function(alpha):
            params = {'alpha': alpha}
            return cross_val_score(Regressor._init_Lasso(params), X=X, y=y, cv=cv_splits,
                                   scoring=scoring, n_jobs=n_jobs).mean()

        parameters = {"alpha": (0.0, 10)}
        return cv_function, parameters


    # -------------- sklearn API functions -------------- #

    def fit(self, X, y, cv_splits=5, scoring='neg_mean_squared_error', n_jobs=-1):
        if self._model_str in ('regressor'):
            models_to_check = ('xgbreg', 'rfr', 'lasso')
            best_model, best_score, best_params = None, None, None
            for model_str in models_to_check:
                print(f'------------------ working on {model_str} ------------------')
                model = Regressor(model_str, self._init_points, self._n_iter)
                cv_function, parameters = model._optimize_function(X, y, cv_splits, scoring, n_jobs)
                best_solution = model._bayesian_optimization(cv_function, parameters)
                params, score = best_solution["params"], best_solution["target"]
                print(f'\tResults for {model_str}:\n\t\tbest params={params}\n\t\tbest score={score}')
                if best_score is None or score > best_score:
                    best_model, best_score, best_params = model, score, params
            best_model._fit(X, y, cv_splits, scoring, n_jobs, best_params)
            self.__dict__.update(best_model.__dict__)
            return self._fitted_model
        else:
            return self._fit(X, y, cv_splits, scoring, n_jobs)

    def _fit(self, X, y, cv_splits=5, scoring='neg_mean_squared_error', n_jobs=-1, params=None):
        if params is None:
            cv_function, parameters = self._optimize_function(X, y, cv_splits, scoring, n_jobs)
            best_solution = self._bayesian_optimization(cv_function, parameters)
            params = best_solution["params"]
        model = self._init_function(params)
        model.fit(X, y)
        self._fitted_model = model
        if self._model_str in ('lr', 'lasso'):
            self.feature_importances_ = self._fitted_model.coef_
        else:
            self.feature_importances_ = self._fitted_model.feature_importances_
        return self._fitted_model

    def predict(self, X):
        if self._fitted_model is not None:
            return self._fitted_model.predict(X)
        else:
            raise Exception('Model should be fitted before prediction')

    def fit_predict(self, X, y, cv_splits=5, scoring='neg_mean_squared_error', n_jobs=-1):
        self.fit(X, y, cv_splits, scoring, n_jobs)
        return self.predict(X)

    def predict_proba(self, X):
        if self._fitted_model is not None:
            return self._fitted_model.predict_proba(X)
        else:
            raise Exception('Model should be fitted before prediction')

    @property
    def feature_importances_(self):
        if self._feature_importances_ is not None:
            return self._feature_importances_
        else:
            raise Exception('model should be fitted before feature_importances_')

    @feature_importances_.setter
    def feature_importances_(self, value):
        self._feature_importances_ = value

    def save_model(self, fname):
        with open(fname, 'wb') as file:
            pickle.dump(self._fitted_model, file)

    def load_model(self, fname):
        with open(fname, 'rb') as file:
            self._fitted_model = pickle.load(file)

    def random_tuning(self, X, y, cv_splits=5, scoring='neg_mean_squared_error', n_jobs=-1, n_iters=100):
        cv_function, parameters = self._optimize_function(X, y, cv_splits, scoring, n_jobs)
        scores = []
        best_score, best_params = -float('inf'), []
        times = []
        t1 = time()
        all_parameters = []
        for param, ranges in parameters.items():
            all_parameters.append(uniform(ranges[0], ranges[-1], n_iters))
        for i in range(n_iters):
            iter_params = [all_parameters[p][i] for p in range(len(all_parameters))]
            score = cv_function(*iter_params)
            if score > best_score:
                best_score = score
                best_params = iter_params
            scores.append(score)
            times.append(time() - t1)
        best_params = {param: best_params[i] for i, param in enumerate(parameters)}
        return scores, times, best_params, best_score, times[-1]

    def greedy_tuning(self, X, y, cv_splits=5, scoring='neg_mean_squared_error', n_jobs=-1, n_iters=100, rounds=3):
        cv_function, parameters = self._optimize_function(X, y, cv_splits, scoring, n_jobs)
        bins = int(n_iters / (rounds * len(parameters)))
        bins = 2 if bins < 2 else bins
        scores = []
        best_score, best_params = 0, {param: (ranges[0] + ranges[-1]) / 2 for param, ranges in parameters.items()}
        times = []
        t1 = time()
        for i in range(rounds):
            for param, ranges in parameters.items():
                best_param_score = -float('inf')
                delta = (ranges[-1] - ranges[0]) / (bins - 1)
                for i in range(bins):
                    current_params = best_params.copy()
                    current_params[param] = ranges[0] + i * delta
                    score = cv_function(*list(current_params.values()))
                    if score > best_param_score:
                        best_param_score = score
                        best_params[param] = current_params[param]
                        if score > best_score:
                            best_score = score
                    scores.append(score)
                    times.append(time() - t1)
        return scores, times, best_params, best_score, times[-1]

    def bayesian_tuning(self, X, y, cv_splits=5, scoring='neg_mean_squared_error', n_jobs=-1, n_iters=100):
        t1 = time()
        cv_function, parameters = self._optimize_function(X, y, cv_splits, scoring, n_jobs)
        gp_params = {"alpha": 1e-5, 'init_points':int(0.3 * n_iters), 'n_iter': int(0.7 * n_iters)}
        bo = BayesianOptimization(cv_function, parameters)
        bo.maximize(**gp_params)
        best_params = bo.max["params"]
        best_score = bo.max["target"]
        scores = list(bo.space.target)
        total_time = time() - t1
        return scores, [total_time/n_iters*i for i in range(n_iters)], best_params, best_score, total_time