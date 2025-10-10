class Solution(object):
    def __init__(self):
        self.result = []
        self.path = []

    def backtracking(self, nums, used):
        if len(self.path) == len(nums):
            self.result.append(self.path[:])
            return

        for i in range(len(nums)):

            if (i > 0 and nums[i] == nums[i - 1] and used[i - 1] == 0) or used[i] == 1:  # * 两个树层去重
                continue

            self.path.append(nums[i])
            used[i] = 1
            self.backtracking(nums, used)
            self.path.pop()
            used[i] = 0

    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        used = [0] * len(nums)

        self.backtracking(sorted(nums), used)

        return self.result
