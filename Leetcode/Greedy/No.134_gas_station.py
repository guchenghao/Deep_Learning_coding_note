class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        result = 0
        current_gain = 0
        sum_gain = 0
        for i in range(len(gas)):
            current_gain += gas[i] - cost[i]
            sum_gain += gas[i] - cost[i]

            if current_gain < 0:
                current_gain = 0
                result = i + 1

        if sum_gain < 0:
            return -1

        return result
