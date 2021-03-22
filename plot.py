import matplotlib.pyplot as plt

def show_plot(x, y, label_x, label_y):
    plt.plot(y, x)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.show()