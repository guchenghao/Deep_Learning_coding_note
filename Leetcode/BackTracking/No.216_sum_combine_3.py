class Solution(object):
    def __init__(self):
        self.result = []
        self.path = []
        self.current_sum = 0

    def backtracking(self, k, n, startindex):
        if self.current_sum > n: # * 剪枝
            return
        if len(self.path) == k and self.current_sum == n:
            self.result.append(self.path[:])
            return

        for i in range(startindex, 9 - (k - len(self.path)) + 2): # * 剪枝
            self.path.append(i)
            self.current_sum += i
            self.backtracking(k, n, i + 1)
            self.current_sum -= i
            self.path.pop()

    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """

        self.backtracking(k, n, 1)

        return self.result





class Solution(object):
    def __init__(self):
        self.result = []
        self.path = []
        self.current_sum = 0

    def backtracking(self, k, n, startindex):
        if self.current_sum >= n and len(self.path) < k: # * 剪枝
            return
        if len(self.path) == k and self.current_sum == n:
            self.result.append(self.path[:])
            return

        for i in range(startindex, 10):
            self.path.append(i)
            self.current_sum += i
            self.backtracking(k, n, i + 1)
            self.current_sum -= i
            self.path.pop()

    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """

        self.backtracking(k, n, 1)

        return self.result
