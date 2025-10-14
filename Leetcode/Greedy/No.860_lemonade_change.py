class Solution(object):
    def lemonadeChange(self, bills):
        """
        :type bills: List[int]
        :rtype: bool
        """
        # * 贪心算法
        # * 优先使用10元找零，再使用5元找零
        charge_5 = 0
        charge_10 = 0

        for bill in bills:

            if bill == 5:
                charge_5 += 1

            elif bill == 10:
                charge_10 += 1
                charge_5 -= 1

            else:
                if charge_10 > 0:
                    charge_10 -= 1
                    charge_5 -= 1

                else:
                    charge_5 -= 3

            if charge_10 < 0 or charge_5 < 0:
                return False

        return True
