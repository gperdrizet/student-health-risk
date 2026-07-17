# Standard library
from __future__ import annotations

# Third party
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, TargetEncoder


def standard_scaler(
        features: list[str],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame = None
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler] | tuple[pd.DataFrame, StandardScaler]:
    '''Standardize continuous features in the given DataFrames using Scikit-learn's StandardScaler.

    Args:
        features (list[str]): List of continuous features to standardize.
        train_df (pd.DataFrame): The training DataFrame.
        test_df (pd.DataFrame, optional): The testing DataFrame. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, StandardScaler] or tuple[pd.DataFrame, StandardScaler]:
        The training and testing dataframes with standardized continuous features and the fitted StandardScaler.
    '''

    scaler = StandardScaler()
    scaler.fit(train_df[features])

    train_df = train_df.copy()
    train_df[features] = scaler.transform(train_df[features])

    if test_df is not None:
        test_df = test_df.copy()
        test_df[features] = scaler.transform(test_df[features])

    if test_df is None:
        return train_df, scaler

    return train_df, test_df, scaler


def add_missing_indicator(
        features: list[str],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame = None
) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    '''Add missing value indicator features to the given DataFrames.

    Args:
        features (list): List of features to create missing value indicators for.
        train_df (pd.DataFrame): The training DataFrame.
        test_df (pd.DataFrame): The testing DataFrame. Default is None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame] or pd.DataFrame: The training and testing DataFrames with added missing value indicator features.
    '''

    indicator_features = [f'{feature}_missing' for feature in features]

    training_missing = train_df[features].isna().astype('int')
    training_missing.columns = indicator_features

    train_df = pd.concat(
        [train_df.reset_index(drop=True),
         training_missing.reset_index(drop=True)],
        axis=1
    )

    if test_df is not None:
        testing_missing = test_df[features].isna().astype('int')
        testing_missing.columns = indicator_features

        test_df = pd.concat(
            [test_df.reset_index(drop=True),
             testing_missing.reset_index(drop=True)],
            axis=1
        )

    if test_df is None:
        return train_df

    return train_df, test_df


def knn_imputer(
        features: list[str],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame = None,
        n_neighbors: int = 5,
        indicator: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    '''Impute missing values in the given features using K-Nearest Neighbors (KNN) imputer.

    Args:
        features (list[str]): The list of features to impute.
        train_df (pd.DataFrame): The training data with missing values.
        test_df (pd.DataFrame): The testing data with missing values.
        n_neighbors (int, optional): Number of neighbors to use for imputation. Defaults to 5.
        indicator (bool, optional): Whether to add a boolean indicator for missing values. Defaults to False.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame] or pd.DataFrame: The training and testing features with missing values imputed.
    '''

    if test_df is not None:
        train_df, test_df, scaler = standard_scaler(features, train_df, test_df)
    else:
        train_df, scaler = standard_scaler(features, train_df)

    if indicator:
        if test_df is not None:
            train_df, test_df = add_missing_indicator(features, train_df, test_df)
        else:
            train_df = add_missing_indicator(features, train_df)

    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')

    train_df = pd.DataFrame(
        imputer.fit_transform(train_df),
        columns=imputer.get_feature_names_out()
    )

    train_df[features] = scaler.inverse_transform(train_df[features])

    if test_df is not None:
        test_df = pd.DataFrame(
            imputer.transform(test_df),
            columns=imputer.get_feature_names_out()
        )
        test_df[features] = scaler.inverse_transform(test_df[features])

    if test_df is None:
        return train_df

    return train_df, test_df


def iterative_imputer(
        features: list[str],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame = None,
        indicator: bool = False,
        max_iter: int = 10,
        initial_strategy: str = 'mean',
        imputation_order: str = 'ascending',
        random_state: int = 315
) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    '''Impute missing values in the given features using Iterative Imputer.

    Args:
        features (list[str]): The list of features to impute.
        train_df (pd.DataFrame): The training data with missing values.
        test_df (pd.DataFrame, optional): The testing data with missing values. Defaults to None.
        indicator (bool, optional): Whether to add a boolean indicator for missing values. Defaults to False.
        max_iter (int, optional): Maximum number of imputation rounds. Defaults to 10.
        initial_strategy (str, optional): Initialization strategy. Defaults to 'mean'.
        imputation_order (str, optional): Feature imputation order. Defaults to 'ascending'.
        random_state (int, optional): Random seed for reproducibility. Defaults to 315.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame] or pd.DataFrame: The training and testing dataframes with missing values imputed.
    '''

    if test_df is not None:
        train_df, test_df, scaler = standard_scaler(features, train_df, test_df)
    else:
        train_df, scaler = standard_scaler(features, train_df)

    if indicator:
        if test_df is not None:
            train_df, test_df = add_missing_indicator(features, train_df, test_df)
        else:
            train_df = add_missing_indicator(features, train_df)

    imputer = IterativeImputer(
        max_iter=max_iter,
        initial_strategy=initial_strategy,
        imputation_order=imputation_order,
        random_state=random_state
    )

    train_df = pd.DataFrame(
        imputer.fit_transform(train_df),
        columns=imputer.get_feature_names_out()
    )

    train_df[features] = scaler.inverse_transform(train_df[features])

    if test_df is not None:
        test_df = pd.DataFrame(
            imputer.transform(test_df),
            columns=imputer.get_feature_names_out()
        )
        test_df[features] = scaler.inverse_transform(test_df[features])

    if test_df is None:
        return train_df

    return train_df, test_df


def one_hot_encoder(
        features: list[str],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    '''Encodes features with Scikit-learn's one-hot-encoder.

    Args:
        features (list[str]): The list of features to encode.
        train_df (pd.DataFrame): The training data to encode.
        test_df (pd.DataFrame, optional): The testing data to encode. Default is None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame] or pd.DataFrame: The training and testing (optional) dataframes with encoded features.
    '''

    feature_encoder = OneHotEncoder(sparse_output=False, drop='first')
    feature_encoder.fit(train_df[features])

    encoded_train_features = feature_encoder.transform(train_df[features])
    encoded_train_df = pd.DataFrame(encoded_train_features, columns=feature_encoder.get_feature_names_out())

    train_df = pd.concat(
        [train_df.drop(columns=features).reset_index(drop=True),
         encoded_train_df.reset_index(drop=True)],
        axis=1
    )

    if test_df is not None:
        encoded_test_features = feature_encoder.transform(test_df[features])
        encoded_test_df = pd.DataFrame(encoded_test_features, columns=feature_encoder.get_feature_names_out())

        test_df = pd.concat(
            [test_df.drop(columns=features).reset_index(drop=True),
             encoded_test_df.reset_index(drop=True)],
            axis=1
        )

    if test_df is None:
        return train_df

    return train_df, test_df


def target_encoder(
        features: list[str],
        train_df: pd.DataFrame,
        train_label: pd.Series,
        test_df: pd.DataFrame = None,
        smooth: float | str = 'auto'
) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    '''Encodes features with Scikit-learn's target encoder.

    Args:
        features (list[str]): The list of features to encode.
        train_df (pd.DataFrame): The training data to encode.
        train_label (pd.Series): The training labels used to fit target encoding.
        test_df (pd.DataFrame, optional): The testing data to encode. Default is None.
        smooth (float|str, optional): Amount of mixing between category and global target mean. Default is 'auto'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame] or pd.DataFrame: Encoded training and optional testing features.
    '''

    feature_encoder = TargetEncoder(smooth=smooth)
    feature_encoder.fit(train_df[features], train_label)

    encoded_train_features = feature_encoder.transform(train_df[features])
    encoded_train_df = pd.DataFrame(encoded_train_features, columns=feature_encoder.get_feature_names_out())

    train_df = pd.concat(
        [train_df.drop(columns=features).reset_index(drop=True),
        encoded_train_df.reset_index(drop=True)],
        axis=1
    )

    if test_df is not None:
        encoded_test_features = feature_encoder.transform(test_df[features])
        encoded_test_df = pd.DataFrame(encoded_test_features, columns=feature_encoder.get_feature_names_out())

        test_df = pd.concat(
            [test_df.drop(columns=features).reset_index(drop=True),
            encoded_test_df.reset_index(drop=True)],
            axis=1
        )

    if test_df is None:
        return train_df

    return train_df, test_df


def encode_label(label: pd.Series) -> tuple[pd.Series, LabelEncoder]:
    '''Encodes the training label with Scikit-learn's label encoder. Returns the encoded
    label as a series and the encoder.

    Args:
        label (pd.DataSeries): The training label column.

    Returns:
        tuple[pd.DataFrame, LabelEncoder]: The encoded labels and the LabelEncoder instance.
    '''

    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(label)

    return label, label_encoder
