class Solution(object):
    def __init__(self):
        self.result = []
        self.path = []
        self.currentsum = 0

    def backtracking(self, candidates, target, startindex):
        if self.currentsum > target:
            return

        if self.currentsum == target:
            # * 超时做法
            if self.path in self.result:
                return
            self.result.append(self.path[:])
            return

        for i in range(startindex, len(candidates)):
            self.path.append(candidates[i])
            self.currentsum += candidates[i]
            self.backtracking(candidates, target, i + 1)  # * 这种startindex的设置其实是一种树枝去重
            self.path.pop()
            self.currentsum -= candidates[i]

    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        can_sorted = sorted(candidates)
        self.backtracking(can_sorted, target, 0)

        return self.result



# * 正解
class Solution(object):
    def __init__(self):
        self.result = []
        self.path = []
        self.currentsum = 0

    def backtracking(self, candidates, target, startindex, used):
        if self.currentsum > target:
            return

        if self.currentsum == target:
            self.result.append(self.path[:])
            return

        for i in range(startindex, len(candidates)):
            if i > 0 and candidates[i] == candidates[i - 1] and used[i - 1] == 0:  # * 树层去重
                continue
            self.path.append(candidates[i])
            self.currentsum += candidates[i]
            used[i] = 1
            self.backtracking(candidates, target, i + 1, used)  # * 树枝去重
            self.path.pop()
            used[i] = 0
            self.currentsum -= candidates[i]

    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        can_sorted = sorted(candidates)  # * 排序是去重的关键
        used = [0] * len(candidates)  # * used数组是去重的关键
        self.backtracking(can_sorted, target, 0, used)

        return self.result
