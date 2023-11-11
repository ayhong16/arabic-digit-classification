from matplotlib import pyplot as plt
from dataparser import DataParser

if __name__ == '__main__':
    parser = DataParser()
    parser.parse_txt("Train_Arabic_Digit.txt", 66)
    for d in range(10):
        plt.figure(d)
        parser.plot_timeseries(d, 1, 3)
    plt.show()
