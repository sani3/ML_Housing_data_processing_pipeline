import matplotlib.pyplot as plt


def print_head(data):
    head = data.head()
    print(head)


def print_info(data):
    info = data.info()
    print(info)


def print_category_levels(data, text_field):
    category_levels = data[text_field].value_counts()
    print(category_levels)


def print_describe(data, num_fields):
    describe = data[num_fields].describe()
    print(describe)


def plot_hist(data, num_fields):
    data[num_fields].hist(bins=50, figsize=(20, 15))
    plt.show()
