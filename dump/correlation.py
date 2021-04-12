from pandas.plotting import scatter_matrix


def co(housing):
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    attributes = ["median_house_value", "median_income", "total_rooms",
                  "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))

    housing.plot(kind="scatter", x="median_income", y="median_house_value",
                 alpha=0.1)
