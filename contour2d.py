from matplotlib import pyplot as plt
from dataparser import DataParser

if __name__ == '__main__':
    parser = DataParser()
    parser.parse_txt("Train_Arabic_Digit.txt", 66)
    comps = [(1, 2), (2, 3), (1, 3)]
    phoneme_map = {
        0: 4,
        1: 4,
        2: 3,
        3: 4,
        4: 3,
        5: 3,
        6: 4,
        7: 3,
        8: 4,
        9: 3,
    }
    for d in range(10):
        parser.plot_gmms(d, comps, phoneme_map[d])
    plt.show()
