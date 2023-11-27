import pandas as pd


class DataParser:

    def __init__(self):
        self.df = pd.DataFrame(columns=['Digit', 'Index', 'Gender', 'MFCCs'])

    def parse_txt(self, filename, num_speakers):
        block_count = 1
        digit = 0
        index = 1
        iterations_per_gender = int((num_speakers / 2) * 10)
        iterations_per_digit = int(num_speakers * 10)
        gender = 'M'
        file = open('../spoken+arabic+digit/' + filename, 'r')
        lines = file.readlines()
        first_line = True
        mfccs = []
        for line in lines:
            if first_line:
                first_line = False
                continue

            stripped = line.strip()
            if len(stripped) == 0:  # new line signifying end of block
                new_row = pd.DataFrame([{'Digit': digit, 'Index': index, 'Gender': gender, 'MFCCs': mfccs}])
                self.df = pd.concat([self.df, new_row], ignore_index=True).reset_index(drop=True)
                mfccs = []
                if block_count % iterations_per_gender == 0:
                    gender = 'F' if gender == 'M' else 'M'
                if block_count % iterations_per_digit == 0:
                    digit += 1
                if index == 10:
                    index = 1
                else:
                    index += 1
                block_count += 1
            else:
                nums = stripped.split(" ")
                mfccs.append([float(num) for num in nums])
        if mfccs:  # If there are remaining MFCCs not yet processed
            new_row = pd.DataFrame([{'Digit': digit, 'Index': index, 'Gender': gender, 'MFCCs': mfccs}])
            self.df = pd.concat([self.df, new_row], ignore_index=True).reset_index(drop=True)

    def get_dataframe(self):
        return self.df
