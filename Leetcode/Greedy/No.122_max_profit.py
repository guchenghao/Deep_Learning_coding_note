class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        # * 贪心算法
        # * 局部最优解
        # * p[3] - p[0] = (p[1]-p[0]) + (p[2]-p[1]) + (p[3]-p[2]), 只要每天的差价是正数就加上
        total_profit = 0

        for i in range(len(prices) - 1):
            profit_everyday = prices[i + 1] - prices[i]

            if profit_everyday > 0:
                total_profit += profit_everyday

        return total_profit
