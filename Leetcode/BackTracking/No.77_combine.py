class Solution(object):
    def backtracking(self, n, k, startindex, path, result):
        if len(path) == k: # * 触发终止条件
            result.append(path[:])
            return

        for i in range(startindex, n + 1):
            path.append(i) # * 做选择
            self.backtracking(n, k, i + 1, path, result) # * 递归
            path.pop() # * 回溯

    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        result = []
        self.backtracking(n, k, 1, [], result)

        return result


class Solution(object):
    def __init__(self):
        self.result = []
        self.path = []

    def backtracking(self, n, k, startindex):
        if len(self.path) == k:
            self.result.append(self.path[:])
            return

        for i in range(startindex, n - (k - len(self.path)) + 2): # * 剪枝 (n- (k - len(self.path)) + 1)
            self.path.append(i)
            self.backtracking(n, k, i + 1)
            self.path.pop()

    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """

        self.backtracking(n, k, 1)

        return self.result
