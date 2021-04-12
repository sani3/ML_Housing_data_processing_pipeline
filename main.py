import os

import numpy as np
from sklearn.pipeline import Pipeline

from transformers import GetData, LoadCsvData, RemoveUnlabelledSamples, ImputMissingNumericValues, CreateNewFeatures, \
    EncodeField, TrainTestSplit, Discretize, XYSplit

num_fields = ['longitude',
              'latitude',
              'housing_median_age',
              'total_rooms',
              'total_bedrooms',
              'population',
              'households',
              'median_income']  # does not include label
text_fields = ['ocean_proximity']  # does not include label
label_field = ['median_house_value']

data_pipeline = Pipeline([
    ("GetData", GetData(
        download_root="https://raw.githubusercontent.com/ageron/handson-ml2/master/",
        data_path=os.path.join("datasets", "housing"),
        data_zip="datasets/housing/housing.tgz"
    )),
    ("LoadCsvData", LoadCsvData(file_name="housing.csv")),
    ("RemoveUnlabelledSamples", RemoveUnlabelledSamples(label_field)),
    ("RemoveUncategorizedSamples", RemoveUnlabelledSamples(text_fields)),
    ("ImputMissingNumericValues", ImputMissingNumericValues(
        num_fields=num_fields, other_fields=text_fields, strategy="median", label_field=label_field
    )),
    ("CreateNewFeatures", CreateNewFeatures(
        col1="rooms_per_household",
        col2="bedrooms_per_room",
        col3="population_per_household"
    )),
    ("EncodeField", EncodeField(fields=text_fields, strategy="onehot")),
    ("Discretize", Discretize(field="median_income",
                              new_discrete_field="income_cat",
                              bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels=[1, 2, 3, 4, 5])
     ),
    ("TrainTestSplit", TrainTestSplit(strat_field="income_cat",
                                      test_size=0.2,
                                      random_state=42,
                                      strategy="strata")
     )
])

train_XYSplit_pipeline = Pipeline([
    ("train_XYSplit", XYSplit(label=label_field))
])

test_XYSplit_pipeline = Pipeline([
    ("test_train", XYSplit(label=label_field))
])

data_train, data_test = data_pipeline.transform(None)

train_X, train_Y = train_XYSplit_pipeline.transform(data_train)

test_X, test_Y = test_XYSplit_pipeline.transform(data_test)


if __name__ == '__main__':
    print('Preprocessing pipeline complete')
