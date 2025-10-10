class Solution(object):
    def __init__(self):
        self.result = []
        self.path = []

    def backtracking(self, nums, used):
        if len(self.path) == len(nums):
            self.result.append(self.path[:])
            return

        for i in range(len(nums)):
            if used[i] == 1:
                continue

            self.path.append(nums[i])
            used[i] = 1
            self.backtracking(nums, used)
            self.path.pop()
            used[i] = 0

    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        used = [0] * len(nums) # * 标记元素是否被使用过

        self.backtracking(nums, used)

        return self.result
