# * 时间复杂度是O(nlogn)
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        num_dict = {}
        for num in nums:
            if num in num_dict:
                num_dict[num] += 1

            else:
                num_dict[num] = 1

        new_num = sorted(num_dict.items(), key=lambda item: item[1], reverse=True)

        return [new_num[i][0] for i in range(k)]


# * 使用最小堆
# * 时间复杂度为 O(nlogk)
import heapq


class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        num_dict = {}
        for num in nums:
            if num in num_dict:
                num_dict[num] += 1

            else:
                num_dict[num] = 1

        min_heap = []

        for item in num_dict.items():
            heapq.heappush(min_heap, (item[1], item[0]))  # * item: (key, freq); min_heap: (freq, key), 依据freq来排序
            if len(min_heap) > k:
                heapq.heappop(min_heap)

        return [heapq.heappop(min_heap)[1] for _ in range(k - 1, -1, -1)]
