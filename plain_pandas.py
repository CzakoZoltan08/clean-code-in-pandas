import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


# Create the DataFrameWrapper class (including the modified get_features method)
class DataFrameWrapper:
    def __init__(self, dataframe: pd.DataFrame):
        self._dataframe = dataframe

    @property
    def sepal_length(self):
        return self._dataframe['sepal length (cm)']

    @sepal_length.setter
    def sepal_length(self, value):
        self._dataframe['sepal length (cm)'] = value

    @property
    def sepal_width(self):
        return self._dataframe['sepal width (cm)']

    @sepal_width.setter
    def sepal_width(self, value):
        self._dataframe['sepal width (cm)'] = value

    @property
    def petal_length(self):
        return self._dataframe['petal length (cm)']

    @petal_length.setter
    def petal_length(self, value):
        self._dataframe['petal length (cm)'] = value

    @property
    def petal_width(self):
        return self._dataframe['petal width (cm)']

    @petal_width.setter
    def petal_width(self, value):
        self._dataframe['petal width (cm)'] = value

    @property
    def target(self):
        return self._dataframe['target']

    @target.setter
    def target(self, value):
        self._dataframe['target'] = value

    def to_dataframe(self):
        return self._dataframe

    def __getattr__(self, attr):
        # Delegate attribute access to the underlying DataFrame
        return getattr(self._dataframe, attr)

    def get_features(self, columns):
        # Extract column names if passed as Series
        column_names = [col.name if isinstance(col, pd.Series) else col for col in columns]
        return self._dataframe[column_names]


def main():
    # Load the Iris dataset from scikit-learn
    iris = load_iris()
    iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_data['target'] = iris.target

    # Wrap the Iris dataset
    wrapped_df = DataFrameWrapper(iris_data)

    # Get features using the property accessors
    features = wrapped_df.get_features([wrapped_df.sepal_length, wrapped_df.petal_length])
    print("Selected features (sepal length and petal length):")
    print(features)

    # Example scikit-learn usage: fitting a logistic regression model
    X = features  # Features from the wrapper
    y = wrapped_df.target  # Target values using the target property

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)
    print("\nPredictions on the Iris dataset:")
    print(predictions)

    # Vectorial operations on properties
    # 1. Addition
    sum_sepal = wrapped_df.sepal_length + wrapped_df.sepal_width
    print("Sum of sepal length and sepal width:")
    print(sum_sepal)

    # 2. Multiplication
    product_petal = wrapped_df.petal_length * wrapped_df.petal_width
    print("\nProduct of petal length and petal width:")
    print(product_petal)

    # 3. Comparison
    is_large_sepal = wrapped_df.sepal_length > 5.0
    print("\nSepal length greater than 5.0:")
    print(is_large_sepal)

    # 4. Combination of operations
    combined_metric = (wrapped_df.sepal_length + wrapped_df.petal_length) / wrapped_df.sepal_width
    print("\nCombined metric ((sepal length + petal length) / sepal width):")
    print(combined_metric)

    # 5. Update property with vectorial operation
    print('\nOriginal sepal lenght values:')
    print(wrapped_df.sepal_length)

    wrapped_df.sepal_length = wrapped_df.sepal_length + 0.5

    print("\nUpdated sepal length after adding 0.5:")
    print(wrapped_df.sepal_length)


if __name__ == "__main__":
    main()