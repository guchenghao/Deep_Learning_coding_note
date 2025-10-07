class Solution(object):
    def __init__(self):
        self.result = []
        self.path = []

    def backtracking(self, nums, startindex):
        if len(self.path) > 1:
            self.result.append(self.path[:])
            # * 注意这里不能return, 因为可能后面还有更长的满足条件的子序列

        uset = set() # * 用于当前树层去重, 因为是从startindex开始的, 所以每一层的startindex都是不同的, 这个uset和used不一样。used是针对每个元素的, 记录这个元素在当前路径中是否使用过
        for i in range(startindex, len(nums)):
            if (self.path and self.path[-1] > nums[i]) or nums[i] in uset:
                continue

            uset.add(nums[i])
            self.path.append(nums[i])
            self.backtracking(nums, i + 1)
            self.path.pop()

    def findSubsequences(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # * 这道题不能排序, 因为排序会改变原数组的相对位置, 影响结果
        self.backtracking(nums, 0)
        return self.result
