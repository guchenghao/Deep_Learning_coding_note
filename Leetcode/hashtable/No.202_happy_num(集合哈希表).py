class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        record = set()
        while n not in record:
            record.add(n)
            new_num = 0
            n_str = str(n)
            for i in n_str:
                new_num+=int(i)**2

            if new_num == 1: 
                return True
            else: 
                n = new_num
        return False