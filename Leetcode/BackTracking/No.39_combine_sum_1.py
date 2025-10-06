class Solution(object):
    def __init__(self):
        self.result = []
        self.path = []
        self.current_sum = 0

    def backtracking(self, target, candidates):
        if self.current_sum > target:
            return

        if self.current_sum == target:
            # * 投机取巧去重
            # * 运行速度较慢
            temp = sorted(self.path[:])
            if temp in self.result:
                return
            self.result.append(temp)
            return

        for num in candidates:
            self.path.append(num)
            self.current_sum += num
            self.backtracking(target, candidates)
            self.path.pop()
            self.current_sum -= num

    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """

        self.backtracking(target, candidates)

        return self.result


# * 正解
# * 回溯模板
class Solution(object):
    def __init__(self):
        self.result = []
        self.path = []
        self.current_sum = 0

    def backtracking(self, target, candidates, startindex):
        if self.current_sum > target:
            return

        if self.current_sum == target:
            self.result.append(self.path[:])
            return

        for i in range(startindex, len(candidates)):
            self.path.append(candidates[i])
            self.current_sum += candidates[i]
            self.backtracking(target, candidates, i)  # * 这里是i不是i+1，因为可以重复选取
            self.path.pop()
            self.current_sum -= candidates[i]

    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """

        self.backtracking(target, candidates, 0)

        return self.result
