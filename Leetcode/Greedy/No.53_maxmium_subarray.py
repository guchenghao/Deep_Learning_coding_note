class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 1:
            return nums[0]
        larger_num = nums[0]
        max_sum = nums[0]

        for i in range(1, len(nums)):
            larger_num = max(nums[i], larger_num + nums[i])
            if larger_num > max_sum:
                max_sum = larger_num

        return max_sum
