class TokenRecord:
    def __init__(self, mfccs, digit, index, gender):
        self.mfccs = mfccs
        self.digit = digit
        self.index = index
        self.gender = gender

    def print_tokens(self):
        print(f'digit: {self.digit}, index: {self.index}, gender: {self.gender}, analysis windows: {len(self.mfccs)}')


l = []

zeros = [token for token in l if token.digit == 0]
