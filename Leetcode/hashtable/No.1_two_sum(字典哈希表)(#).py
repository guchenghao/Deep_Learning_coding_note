class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        nums_dict = {}
        result = []

        for i in range(len(nums)):
            if nums[i] not in nums_dict:
                nums_dict[target - nums[i]] = [i]
            
            else:
                result = nums_dict[nums[i]] + [i]
                break

        return result