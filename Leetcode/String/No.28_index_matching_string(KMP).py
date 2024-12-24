class Solution(object):
    # * 计算KMP算法中前缀数组
    def getPrefix_arr(self, next, mode_str):
        j = 0  # * 前缀末尾
        next[0] = 0

        # * i 表示后缀末尾
        for i in range(1, len(mode_str)):
            while j > 0 and mode_str[i] != mode_str[j]:
                j = next[j - 1]

            if mode_str[i] == mode_str[j]:
                j += 1

            next[i] = j

    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        next = [0] * len(needle)

        self.getPrefix_arr(next, needle)

        j = 0

        for i in range(len(haystack)):
            while j > 0 and needle[j] != haystack[i]:
                j = next[j - 1]

            if needle[j] == haystack[i]:
                j += 1

            if j == len(needle):
                return i - len(needle) + 1

        return -1
