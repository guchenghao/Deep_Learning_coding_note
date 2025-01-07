# * 这道题需要自行构建单调队列
# * 这道题很经典，要记住！
# * 需要用deque这个数据结构，用list会超时
from collections import deque


class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        queue_mono = deque()  # * 自己构建单调队列，这个单调指的是单调递增或者单调递减
        result = []

        for i in range(k):
            if not queue_mono:
                queue_mono.append(nums[i])

            else:
                while queue_mono and queue_mono[-1] < nums[i]:
                    queue_mono.pop()

                queue_mono.append(nums[i])

        result.append(queue_mono[0])

        for j in range(k, len(nums)):
            
            # * 其实这个操作是为了判断当前滑动窗口需要pop的元素是否是最大值，如果不是的话，因为push操作已经pop了一些较小的值，所以不需要对单调队列进行pop操作
            if queue_mono[0] == nums[j - k]:  # * 这个点很容易出错，在pop的时候需要格外注意
                queue_mono.popleft()

            while queue_mono and queue_mono[-1] < nums[j]:
                queue_mono.pop()

            queue_mono.append(nums[j])

            result.append(queue_mono[0])

        return result
