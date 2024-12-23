
# * 这道题的思路与No.27的思路基本一致
# * 这个解法(solution1)没什么太大的意义
class Solution1(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        # * 大体思路还是采用双指针的方式
        arr_s = s.split()

        left = 0
        right = len(arr_s) - 1

        print(arr_s)
        while left < right:
            if arr_s[left] == " ":
                left += 1
                continue

            if arr_s[right] == " ":
                right -= 1
                continue

            temp = arr_s[left]
            arr_s[left] = arr_s[right]
            arr_s[right] = temp

            left += 1
            right -= 1

        return " ".join(arr_s)
