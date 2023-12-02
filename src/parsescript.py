from dataparser import DataParser

if __name__ == "__main__":
    train_parser = DataParser()
    train_parser.parse_txt("Train_Arabic_Digit.txt", 66)
    train_df = train_parser.get_dataframe()
    test_parser = DataParser()
    test_parser.parse_txt("Test_Arabic_Digit.txt", 22)
    test_df = test_parser.get_dataframe()
    train_df.to_csv("../data/train_df.csv", index=False)
    test_df.to_csv("../data/test_df.csv", index=False)

