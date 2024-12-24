class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # * 这道题之所以不推荐用哈希表，是因为hash方法太复杂了
        # * a + b + c = 0, 固定a，移动b和c
        result = []
        nums_sorted = sorted(nums)  # * 因为这道题最终解决不需要返回与index相关的值，所以可以排序

        for i in range(len(nums)):
            if nums_sorted[i] > 0:
                return result
            if i > 0 and nums_sorted[i] == nums_sorted[i - 1]:  # * 去重的第一个关键点: 之所以这里是i - 1，是因为数组已经经过排序了
                continue
            left = i + 1
            right = len(nums) - 1

            a = nums_sorted[i]

            while left < right:
                b = nums_sorted[left]
                c = nums_sorted[right]
                if a + b + c == 0:
                    result.append([a, b, c])

                    # *  跳过相同的元素以避免重复 (关键点)
                    while right > left and nums_sorted[right] == nums_sorted[right - 1]:
                        right -= 1
                    while right > left and nums_sorted[left] == nums_sorted[left + 1]:
                        left += 1

                    right -= 1
                    left += 1

                elif a + b + c < 0:
                    left += 1

                else:
                    right -= 1

        return result
