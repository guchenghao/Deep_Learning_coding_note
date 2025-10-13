class Solution(object):
    # * 跳跃游戏可以采用回溯法，但是效率不高，会超时
    # * 关心路径
    def __init__(self):
        self.current_pos = 1

    def backtracking(self, step, nums):
        if self.current_pos == len(nums):
            return True

        for i in range(1, step + 1):
            self.current_pos += i
            if self.backtracking(nums[self.current_pos - 1], nums):
                return True
            self.current_pos -= i

        return False

    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """

        return self.backtracking(nums[0], nums)


class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # * 贪心算法
        # * 只考虑每步的最大跳跃范围， 不考虑具体跳跃路径
        if len(nums) == 1:
            return True
        cover = 0

        for i in range(len(nums)):
            if i <= cover:
                cover = max(cover, nums[i] + i)

                if cover >= len(nums) - 1:
                    return True

        return False
