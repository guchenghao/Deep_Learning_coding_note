class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        result = []
        nums_sorted = sorted(
            nums
        )  # * 因为这道题最终解决不需要返回与index相关的值，所以可以排序

        for k in range(len(nums)):
            # * 剪枝
            if nums_sorted[k] > target and target > 0:
                break

            if k > 0 and nums_sorted[k] == nums_sorted[k - 1]:
                continue

            a = nums_sorted[k]

            # * 以下所有代码均与三数之和一致
            for i in range(k + 1, len(nums)):
                b = nums_sorted[i]
                
                if a + b > target and target > 0:
                    break
                left = i + 1
                right = len(nums) - 1

                if i > k + 1 and nums_sorted[i - 1] == b:  # * 确保相同的4元组解法有一种
                    continue

                while left < right:
                    c = nums_sorted[left]
                    d = nums_sorted[right]
                    if a + b + c + d == target:
                        result.append([a, b, c, d])
                        
                        # *  跳过相同的元素以避免重复 (关键点)
                        while (
                            right > left
                            and nums_sorted[right] == nums_sorted[right - 1]
                        ):
                            right -= 1
                        while (
                            right > left and nums_sorted[left] == nums_sorted[left + 1]
                        ):
                            left += 1

                        right -= 1
                        left += 1

                    elif a + b + c + d < target:
                        left += 1

                    else:
                        right -= 1

        return result
