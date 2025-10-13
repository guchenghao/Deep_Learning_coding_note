class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # * 贪心算法
        # * 只考虑每步的最大跳跃范围， 不考虑具体跳跃路径
        # * 只考虑覆盖范围
        if len(nums) == 1:
            return 0
        cover = 0
        next_jump = 0
        step = 0
        i = 0
        for i in range(len(nums)):
            next_jump = max(next_jump, nums[i] + i)
            if i == cover:  # * 当前步的最大跳跃范围，每步最大跳跃范围的边界
                step += 1
                cover = next_jump
                if cover >= len(nums) - 1:  # * 如果已经可以到达终点
                    break

        return step
