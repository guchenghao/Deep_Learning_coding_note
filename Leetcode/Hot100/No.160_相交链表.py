# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        curA, curB = headA, headB

        len_A = 0
        len_B = 0

        while curA:
            len_A += 1
            curA = curA.next

        while curB:
            len_B += 1
            curB =curB.next
        
        curA, curB = headA, headB

        if len_A > len_B:
            while curA and curB:
                if curA == curB:
                    return curA
                else:
                    if len_A > len_B:
                        curA = curA.next
                        len_A -= 1
                    else:
                        curA = curA.next
                        curB = curB.next
                        len_A -= 1
                        len_B -= 1
            
        else:
            while curA and curB:
                if curA == curB:
                    return curA
                else:
                    if len_A < len_B:
                        curB = curB.next
                        len_B -= 1
                    else:
                        curA = curA.next
                        curB = curB.next
                        len_A -= 1
                        len_B -= 1
        

        return None