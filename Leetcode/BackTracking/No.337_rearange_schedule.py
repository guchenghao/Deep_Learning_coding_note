from collections import defaultdict


class Solution(object):
    # * 这道题严格意义上说不是回溯算法，而是欧拉路径问题
    # * 但可以用回溯算法来做
    def backtracking(self, airport, targets, result):
        while targets[airport]:  # * 当机场还有可到达的机场时
            next_airport = targets[airport].pop()  # * 弹出下一个机场
            self.backtracking(
                next_airport, targets, result
            )  # * 递归调用回溯函数进行深度优先搜索
        result.append(airport)  # * 将当前机场添加到行程路径中, 因为是后序添加, 所以最终需要逆序返回

    def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """
        self.tickets_dict = defaultdict(list)

        for ticket in tickets:
            self.tickets_dict[ticket[0]].append(ticket[1])

        for key in self.tickets_dict:
            self.tickets_dict[key].sort(reverse=True)  # * 对到达机场列表进行字母逆序排序, 以便后续使用pop()时能按字母顺序获取最小的机场

        result = []

        self.backtracking("JFK", self.tickets_dict, result)  # * 调用回溯函数开始搜索路径
        return result[::-1]  # * 返回逆序的行程路径, 因为路径是从终点到起点添加的
