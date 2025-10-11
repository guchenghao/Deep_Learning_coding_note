class Solution(object):
    def findContentChildren(self, g, s):
        """
        :type g: List[int]
        :type s: List[int]
        :rtype: int
        """
        i = 0
        j = 0
        result = 0
        s.sort()
        g.sort()
        while i < len(g) and j < len(s):
            if s[j] >= g[i]:
                result += 1
                i += 1
                j += 1

            else:
                j += 1

        return result
