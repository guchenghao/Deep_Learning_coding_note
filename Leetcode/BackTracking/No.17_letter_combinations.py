class Solution(object):
    def __init__(self):
        self.nums_letters = {
            "2": ["a", "b", "c"],
            "3": ["d", "e", "f"],
            "4": ["g", "h", "i"],
            "5": ["j", "k", "l"],
            "6": ["m", "n", "o"],
            "7": ["p", "q", "r", "s"],
            "8": ["t", "u", "v"],
            "9": ["w", "x", "y", "z"],
        }
        self.path = []
        self.result = []

    def backtracking(self, digits, startindex):
        if len(self.path) == len(digits):
            self.result.append("".join(self.path))
            return

        # * 双重for循环的回溯
        for i in range(startindex, len(digits)):
            current_letter = digits[i]
            for letter in self.nums_letters[current_letter]:
                self.path.append(letter)
                self.backtracking(digits, i + 1)
                self.path.pop()

    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return self.result

        self.backtracking(digits, 0)

        return self.result


class Solution(object):
    def __init__(self):
        self.nums_letters = {
            "2": ["a", "b", "c"],
            "3": ["d", "e", "f"],
            "4": ["g", "h", "i"],
            "5": ["j", "k", "l"],
            "6": ["m", "n", "o"],
            "7": ["p", "q", "r", "s"],
            "8": ["t", "u", "v"],
            "9": ["w", "x", "y", "z"],
        }
        self.path = []
        self.result = []

    def backtracking(self, digits, startindex):
        if len(self.path) == len(digits):
            self.result.append("".join(self.path))
            return

        current_letter = digits[startindex]
        for letter in self.nums_letters[current_letter]:
            self.path.append(letter)
            self.backtracking(digits, startindex + 1)
            self.path.pop()

    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return self.result

        self.backtracking(digits, 0)

        return self.result
