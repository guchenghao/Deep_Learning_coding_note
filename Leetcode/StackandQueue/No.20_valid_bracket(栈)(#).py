class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # * 这道题是很经典的利用栈来解决的问题
        stack = []

        arr_s = list(s)
        
        # * 之所以这里在遇到左括号的时候，压入右括号，是为了方便进行比较
        # * 直接压入左括号也可以，只是这样在进行比较的时候，需要进行括号类型比较，代码会复杂很多
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
