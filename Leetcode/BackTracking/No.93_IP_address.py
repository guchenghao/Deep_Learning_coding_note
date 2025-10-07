class Solution(object):
    def __init__(self):
        self.result = []
        self.path = []
        self.current_len = 0  # * 记录当前path中数字的长度和, 用于判断是否使用了s中的所有字符

    def is_IP(self, s_arr):
        for num in s_arr:
            if num[0] == "0" and len(num) > 1:
                return False     
        return True

    def backtracking(self, s, startindex):
        if len(self.path) == 4:
            if self.is_IP(self.path) and self.current_len == len(s):
                self.result.append(".".join(self.path[:]))
            return

        for i in range(startindex, len(s)):
            if int(s[startindex:i + 1]) >= 0 and int(s[startindex:i + 1]) <= 255:
                self.path.append(s[startindex:i + 1])
                self.current_len += len((s[startindex:i + 1]))
                self.backtracking(s, i + 1)
                self.path.pop()
                self.current_len -= len((s[startindex:i + 1]))
            else:
                continue

    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """

        self.backtracking(s, 0)

        return self.result


# * 优化版本, 去掉current_len, 在path长度为3时直接判断剩余字符串是否合法
# * 只需要切割3次
class Solution(object):
    def __init__(self):
        self.result = []
        self.path = []

    def is_IP(self, s_arr):
        for num in s_arr:
            if not num:
                return False
            if num[0] == "0" and len(num) > 1:
                return False

            if int(num) >= 0 and int(num) <= 255:
                continue
            else:
                return False
        return True

    def backtracking(self, s, startindex):
        if len(self.path) == 3:  # * 当path中已经有3个数字时, 直接判断剩余字符串是否合法
            self.path.append(s[startindex:])
            if self.is_IP(self.path):
                self.result.append(".".join(self.path[:]))

            self.path.pop()
            return

        for i in range(startindex, len(s)):
            if int(s[startindex : i + 1]) >= 0 and int(s[startindex : i + 1]) <= 255:
                self.path.append(s[startindex : i + 1])
                self.backtracking(s, i + 1)
                self.path.pop()
            else:
                continue

    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """

        self.backtracking(s, 0)

        return self.result
