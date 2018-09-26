import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, cross_val_score


def nulls_summary(df: pd.DataFrame):
    nulls_summary = pd.DataFrame(df.isnull().any(), columns=['Nulls'])
    nulls_summary['Num_of_nulls [qty]'] = pd.DataFrame(df.isnull().sum())
    nulls_summary['Num_of_nulls [%]'] = round((df.isnull().mean() * 100), 2)
    print(nulls_summary)


def outliers(df: pd.DataFrame):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    low_boundary = (q1 - 1.5 * iqr)
    upp_boundary = (q3 + 1.5 * iqr)
    num_of_out_L = (df[iqr.index] < low_boundary).sum()
    num_of_out_U = (df[iqr.index] > upp_boundary).sum()
    outliers = pd.DataFrame({'lower_value': low_boundary, 'upper_boundary': upp_boundary, 'num_of_outliers_L':num_of_out_L, 'num_of_outliers_U':num_of_out_U})
    print(outliers)


def value_counts(df: pd.DataFrame):
    for col in df.select_dtypes(['object', 'category']):
        print(df[col].value_counts())


# def numerical_distribution(df: pd.DataFrame):
#     df.hist(bins=12)  # histogram dla wszystkich zmiennych
#     from scipy import stats  # test na normalność rozkładu
#     df.select_dtypes([float, int]).apply(stats.normaltest)  # p-value to wartość

def get_dummies(df: pd.DataFrame, col_name, drop_col=True) -> pd.DataFrame:
    dummies = pd.get_dummies(df[col_name], prefix=col_name, drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    if drop_col:
        df.drop(col_name, inplace=True, axis=1)
    return df


def scale(df: pd.DataFrame, col_name, scaler='minMax') -> pd.DataFrame:
    scalers = {
        'standard': StandardScaler,
        'minMax': MinMaxScaler
    }
    df[col_name] = scalers[scaler]().fit_transform(pd.DataFrame(df[col_name])).flatten().tolist()
    return df


def fillna(df: pd.DataFrame, col_name, func='mean') -> pd.DataFrame:
    functions = {
        'mean': lambda x: np.mean(x),
        'median': lambda x: np.median(x)
    }
    df[col_name] = df.fillna(functions[func](df[col_name]))

    return df


def cat_to_nums(df: pd.DataFrame, col_name, ordered_vals) -> pd.DataFrame:
    for i, val in enumerate(ordered_vals):
        try:
            df.at[df.loc[df[col_name] == val].index, col_name] = i
        except TypeError:
            continue
    return df


def train_models(_models, X, y, scoring='accuracy', reversed_scoring=False):
    models = {}
    for model in _models:
        model_name = get_model_name(model)
        models[model_name] = model

    model_scores = {}
    kf = KFold(5, True, 5)
    for model_name, model in models.items():
        scores = []
        for k, (train, test) in enumerate(kf.split(X, y)):
            model.fit(X[train], y[train])
            score = model.score(X[test], y[test])
            scores.append(score)

        results = cross_val_score(model, X, y, cv=kf, scoring=scoring)

        model_score = results.mean()
        model_scores[model_name] = model_score
        # print(model_name, str(model_score))

    sorted_model_scores = sorted(model_scores, key=model_scores.get, reverse=True)

    if reversed_scoring:
        sorted_model_scores = reversed(sorted_model_scores)

    sorted_scores = [(k, model_scores[k]) for k in sorted_model_scores]

    best_model_name = None
    for k, v in sorted_scores:
        if best_model_name is None:
            best_model_name = k
        print(k, v)

    return models[best_model_name]


def get_model_name(model):
    return str(type(model)).rsplit('.', 1)[-1][:-2]


def list_diff(first, second):
    second = set(second)
    first = set(first)
    return [item for item in first if item not in second]
