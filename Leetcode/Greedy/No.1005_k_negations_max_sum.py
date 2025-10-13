class Solution(object):
    def largestSumAfterKNegations(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        # * 贪心算法
        # * 每次都将当前数组中最小的数取反
        # * 直到取反k次
        for _ in range(k):
            nums.sort()
            nums[0] = -nums[0]

        return sum(nums)


class Solution(object):
    def largestSumAfterKNegations(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        # * 贪心算法
        # * 先将数组按绝对值从大到小排序
        # * 两次贪心
        nums.sort(key=lambda x: abs(x), reverse=True)  # 第一步：按照绝对值降序排序数组A

        for i in range(len(nums)):  # 第二步：执行K次取反操作
            if nums[i] < 0 and k > 0:
                nums[i] *= -1
                k -= 1

        if k % 2 == 1:  # 第三步：如果K还有剩余次数，将绝对值最小的元素取反
            nums[-1] *= -1

        result = sum(nums)  # 第四步：计算数组A的元素和
        return result
