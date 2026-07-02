# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: bool
        """
        # * 暴力解法
        res = []
        
        while head:
            res.append(head.val)
            head = head.next
        
        # * 双指针法
        while len(res) > 1:
            if res.pop(0) == res.pop():
                continue
            else:
                return False
        
        return True




