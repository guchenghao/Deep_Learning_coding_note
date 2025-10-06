class Solution(object):
    def __init__(self):
        self.result = []
        self.path = []

    def palindrome(self, s):
        if not s:
            return False
        if len(s) == 1:
            return True

        i = 0
        j = len(s) - 1

        while i < j:
            if s[i] == s[j]:
                i += 1
                j -= 1
            else:
                return False

        return True

    def backtracking(self, s, startindex):
        # * 回溯的过程中，startindex代表的是当前回溯到字符串的哪个位置了
        if startindex == len(s):
            self.result.append(self.path[:])
            return

        for i in range(startindex, len(s)):
            if not self.palindrome(s[startindex : i + 1]):
                continue
            else:
                self.path.append(s[startindex : i + 1])
                self.backtracking(s, i + 1)
                self.path.pop()

    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """

        self.backtracking(s, 0)

        return self.result
