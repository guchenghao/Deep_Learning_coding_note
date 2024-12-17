
# * 大体思路就是将4数之和拆为两数之和进行求解，思路和两数之和完全一致
# * 时间复杂度从O(n ** 4) -> O(n ** 2)
# * a + b + c + d = (a + b) + (c + d)
class Solution(object):
    def fourSumCount(self, nums1, nums2, nums3, nums4):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :type nums3: List[int]
        :type nums4: List[int]
        :rtype: int
        """
        count = 0
        nums_dict = {}
        for i in range(len(nums1)):
            for j in range(len(nums2)):
                sum_1 = nums1[i] + nums2[j]
                if -sum_1 not in nums_dict:
                    nums_dict[-sum_1] = 1

                else:
                    nums_dict[-sum_1] += 1

        for p in range(len(nums3)):
            for q in range(len(nums4)):
                sum_2 = nums3[p] + nums4[q]

                if sum_2 in nums_dict:
                    count += nums_dict[sum_2]

        return count
