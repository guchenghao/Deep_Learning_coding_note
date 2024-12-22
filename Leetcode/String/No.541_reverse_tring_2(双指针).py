class Solution(object):
    def reverseStr1(self, arr_s):
        left = 0
        right = len(arr_s) - 1

        while left < right:
            temp = arr_s[left]
            arr_s[left] = arr_s[right]
            arr_s[right] = temp

            left += 1
            right -= 1

        return arr_s

    def reverseStr(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        # * 总体思路基于reverse string 1的题目
        # * 主要使用python数组的切片操作
        # * 这道题重点在于如何处理不同长度下的字符串翻转，算是一道模拟题
        if k == 1:
            return s
        arr_s = list(s)
        str_len = len(arr_s)

        if str_len < k:
            result = self.reverseStr1(arr_s)

        elif str_len >= k and str_len <= 2 * k:
            result = self.reverseStr1(arr_s[:k])

            result = result + arr_s[k:]

        else:
            mod = str_len % (2 * k)
            turn = str_len // (2 * k)
            result = []

            for i in range(turn):
                temp = arr_s[2 * k * i : 2 * k * (i + 1)]
                result += self.reverseStr1(temp[:k])
                result += temp[k:]

            if mod < k:
                temp = arr_s[turn * 2 * k :]
                result += self.reverseStr1(temp)
            elif mod >= k and mod < 2 * k:
                temp = arr_s[turn * 2 * k :]
                result += self.reverseStr1(temp[:k])
                result += temp[k:]

        return "".join(result)
