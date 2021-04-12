import os
import tarfile
import pandas as pd
from six.moves import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


# Custom transformer to download data to data directory and extract to a new directory
class GetData(BaseEstimator, TransformerMixin):
    def __init__(self,
                 download_root="https://raw.githubusercontent.com/ageron/handson-ml2/master/",
                 data_path=os.path.join("datasets", "housing"),
                 data_zip="datasets/housing/housing.tgz"
                 ):  # no *args or **kargs
        self.download_root = download_root
        self.data_path = data_path
        self.data_url = self.download_root + data_zip

    def fit(self, x, y=None):
        return self  # nothing else to do

    def transform(self, x=None, y=None, ):  # no *args or **kargs
        if not os.path.isdir(self.data_path):
            os.makedirs(self.data_path)
        tgz_path = os.path.join(self.data_path, "housing.tgz")
        urllib.request.urlretrieve(self.data_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=self.data_path)
        housing_tgz.close()


# Custom transformer to load data to DataFrame
class LoadCsvData(BaseEstimator, TransformerMixin):
    def __init__(self, file_name="housing.csv"):  # no *args or **kargs
        self.file_name = file_name

    def fit(self, x, y=None):
        return self  # nothing else to do

    def transform(self, x, y=None):  # no *args or **kargs
        path = os.path.join("datasets", "housing")
        file_path = os.path.join(path, self.file_name)
        return pd.read_csv(file_path)


# Custom transformer for  numeric feature scaling
class ScaleNumFields(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_fields, strategy):  # no *args or **kargs
        self.numeric_fields = numeric_fields
        self.strategy = strategy

    def fit(self, x, y=None):
        return self  # nothing else to do

    def transform(self, x, y=None):
        for field in self.numeric_fields:
            if self.strategy == "min-max":  # min-max scaling also normalization
                x[field] = (x[field] - x[field].min(axis=0)) / (x[field].max(axis=0) - x[field].min(axis=0))
                return x
            if self.strategy == "standard":  # Standardization
                x[field] = (x[field] - x[field].mean(axis=0)) / x[field].std(axis=0)
            return x


# Custom transformer remove unlabelled samples or samples with missing value in category fields
class RemoveUnlabelledSamples(BaseEstimator, TransformerMixin):
    def __init__(self, label_field="median_house_value"):  # no *args or **kargs
        self.label_field = label_field

    def fit(self, x, y=None):
        return self  # nothing else to do

    def transform(self, x, y=None):
        # remove unlabelled samples
        data = x.dropna(
            axis=0, how='any', thresh=None, subset=self.label_field, inplace=False
        )
        data.index = range(data.shape[0])
        return data


# Custom transformer remove samples with Na values
class RemoveSamplesWithNa(BaseEstimator, TransformerMixin):
    def __init__(self, label_field="median_house_value"):  # no *args or **kargs
        self.label_field = label_field

    def fit(self, x, y=None):
        return self  # nothing else to do

    def transform(self, x, y=None):
        # remove samples with Na values
        data = x.dropna(
            axis=0, how='any', thresh=None, subset=None, inplace=False
        )
        data.index = range(data.shape[0])
        return data


# Custom transformer remove field
class RemoveField(BaseEstimator, TransformerMixin):
    def __init__(self, field="median_house_value"):  # no *args or **kargs
        self.field = field  # field is a list of str

    def fit(self, x, y=None):
        return self  # nothing else to do

    def transform(self, x, y=None):
        data = x.drop(
            labels=self.field, axis=1, index=None, columns=None,
            level=None, inplace=False, errors='raise'
        )
        return data


# Custom transformer to  discretize continuous numeric feature
class Discretize(BaseEstimator, TransformerMixin):
    def __init__(self,
                 field,  # ="median_income",
                 new_discrete_field,  # ="income_cat",
                 bins,  # =[0., 1.5, 3.0, 4.5, 6., np.inf],
                 labels,  # =[1, 2, 3, 4, 5]
                 ):  # no *args or **kargs
        self.field = field
        self.new_discrete_field = new_discrete_field
        self.bins = bins
        self.labels = labels

    def fit(self, x, y=None):
        return self  # nothing else to do

    def transform(self, x, y=None):
        x[self.new_discrete_field] = pd.cut(x[self.field], bins=self.bins, labels=self.labels)
        x.drop(
            labels=self.field, axis=1, index=None, columns=None,
            level=None, inplace=True, errors='raise'
        )
        return x


# Custom transformer to  split data to train and test sets
class TrainTestSplit(BaseEstimator, TransformerMixin):
    def __init__(self, strat_field, test_size=0.2, random_state=42, strategy="random"):  # no *args or **kargs
        self.strat_field = strat_field
        self.test_size = test_size
        self.random_state = random_state
        self.strategy = strategy

    def fit(self, x, y=None):
        return self  # nothing else to do

    def transform(self, x, y=None):
        if self.strategy == "random":
            train_set, test_set = train_test_split(x, test_size=self.test_size, random_state=self.random_state)
            return train_set, test_set
        if self.strategy == "strata":
            split = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
            for train_index, test_index in split.split(x, x[self.strat_field]):
                strat_train_set = x.loc[train_index]
                strat_test_set = x.loc[test_index]

                # for set_ in (strat_train_set, strat_test_set):
                #     set_.drop(self.strat_field, axis=1, inplace=True)

                return strat_train_set, strat_test_set


# Custom transformer to create and add new features
class CreateNewFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, col1, col2, col3):  # no *args or **kargs
        self.col1 = col1
        self.col2 = col2
        self.col3 = col3

    def fit(self, x, y=None):
        return self  # nothing else to do

    def transform(self, x, y=None):
        x[self.col1] = x["total_rooms"] / x["households"]
        x[self.col2] = x["total_bedrooms"] / x["total_rooms"]
        x[self.col3] = x["population"] / x["households"]
        return x


# Custom transformer to input missing numeric values
class ImputMissingNumericValues(BaseEstimator, TransformerMixin):
    def __init__(self, num_fields, label_field, other_fields, strategy="median"):  # no *args or **kargs
        self.num_fields = num_fields
        self.other_fields = other_fields
        self.strategy = strategy
        self.label_field = label_field

    def fit(self, x, y=None):
        return self  # nothing else to do

    def transform(self, x, y=None):
        imputer = SimpleImputer(strategy=self.strategy)
        imputer.fit(x[self.num_fields])
        x_num = imputer.transform(x[self.num_fields])
        x_num = pd.DataFrame(x_num, columns=self.num_fields)
        return pd.concat([x_num, x[self.other_fields], x[self.label_field]], axis=1)


# Custom transformer to encode levels of category field
class EncodeField(BaseEstimator, TransformerMixin):
    def __init__(self, fields, strategy="ordinal"):  # no *args or **kargs
        self.fields = fields
        self.strategy = strategy

    def fit(self, x, y=None):
        return self  # nothing else to do

    def transform(self, x, y=None):
        if self.strategy == "ordinal":
            ordinal_encoder = OrdinalEncoder()
            x_code = ordinal_encoder.fit_transform(x[self.fields])
            return pd.concat([x.drop(labels=self.fields, axis=1), pd.DataFrame(x_code, columns=self.fields)], axis=1)
        if self.strategy == "onehot":
            onehot_encoder = OneHotEncoder()
            x_code = onehot_encoder.fit_transform(x[self.fields])
            return pd.concat([x.drop(labels=self.fields, axis=1),
                              pd.DataFrame(x_code.toarray(), columns=onehot_encoder.categories_)],
                             axis=1
                             )


# Custom transformer to split data to x and y label
class XYSplit(BaseEstimator, TransformerMixin):
    def __init__(self, label):  # no *args or **kargs
        self.label = label

    def fit(self, x, y=None):
        return self  # nothing else to do

    def transform(self, x, y=None):
        xx = x.drop(self.label, axis=1)
        yy = x[self.label].copy()
        return xx, yy

