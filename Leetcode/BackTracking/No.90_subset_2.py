class Solution(object):
    def __init__(self):
        self.result = [[]]
        self.path = []

    def backtracking(self, nums, startindex):
        for i in range(startindex, len(nums)):
            self.path.append(nums[i])
            if self.path[:] not in self.result:
                self.result.append(self.path[:])
            self.backtracking(nums, i + 1)
            self.path.pop()

    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # * 先排序, 这样相同的元素就会相邻
        # * 排序对于回溯算法中去重是一个常用的技巧
        self.backtracking(sorted(nums), 0)

        return self.result





class Solution(object):
    def __init__(self):
        self.result = [[]]
        self.path = []

    def backtracking(self, nums, startindex, used):
        for i in range(startindex, len(nums)):
            if i > 0 and nums[i] == nums[i - 1] and used[i - 1] == 0:  # * 树层去重
                continue

            self.path.append(nums[i])
            used[i] = 1
            self.result.append(self.path[:])
            self.backtracking(nums, i + 1, used)  # * 树枝去重
            used[i] = 0
            self.path.pop()

    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        used = [0] * len(nums)
        self.backtracking(sorted(nums), 0, used)

        return self.result
