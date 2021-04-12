import matplotlib.pyplot as plt


def vis1(data):
    data.plot(kind="scatter", x="longitude", y="latitude")


def vis2(data):
    data.plot(
        kind="scatter", x="longitude", y="latitude", alpha=0.4,
        s=data["population"] / 100, label="population", figsize=(10, 7),
        c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    )
    plt.legend()


# Consider removing this file instead, call plotting functions directly in main
