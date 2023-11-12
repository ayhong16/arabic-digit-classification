from dataparser import DataParser

if __name__ == '__main__':
    parser = DataParser()
    parser.parse_txt("Train_Arabic_Digit.txt", 66)
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
    digit = 4
    n_components = phoneme_map[digit]
    gmm_params = parser.compute_gmm_params(digit, n_components, True, "full")
    parser.plot_pdfs(gmm_params)
