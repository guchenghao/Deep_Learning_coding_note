class Solution(object):
    def removeDuplicates(self, s):
        """
        :type s: str
        :rtype: str
        """
        stack = []

        for char in s:
            if not stack:
                stack.append(char)
                continue

            if stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)

        return "".join(stack)
