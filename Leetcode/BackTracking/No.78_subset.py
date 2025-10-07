class Solution(object):
    def __init__(self):
        self.result = [[]]
        self.path = []

    def backtracking(self, nums, startindex):
        for i in range(startindex, len(nums)):
            self.path.append(nums[i])
            self.result.append(self.path[:])
            self.backtracking(nums, i + 1)
            self.path.pop()

    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        self.backtracking(nums, 0)

        return self.result


class Solution(object):
    def __init__(self):
        self.result = []
        self.path = []

    def backtracking(self, nums, startindex):
        self.result.append(self.path[:])  # * 每次进入递归函数就添加当前path
        if startindex >= len(nums):  # * 递归终止条件
            return
        for i in range(startindex, len(nums)):
            self.path.append(nums[i])
            self.backtracking(nums, i + 1)
            self.path.pop()

    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        self.backtracking(nums, 0)

        return self.result
