class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # * 这道题是很经典的利用栈来解决的问题
        stack = []

        arr_s = list(s)

        for i in range(len(arr_s)):
            if arr_s[i] == "(":
                stack.append(")")

            elif arr_s[i] == "{":
                stack.append("}")

            elif arr_s[i] == "[":
                stack.append("]")

            else:
                if not stack or stack[-1] != arr_s[i]:
                    return False

                stack.pop()

        return True if not stack else False
