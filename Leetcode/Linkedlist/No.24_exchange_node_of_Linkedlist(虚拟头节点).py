class Solution(object):
    def swapPairs(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        dummy_head = ListNode(next=head)

        cur = dummy_head

        while cur.next and cur.next.next:
            temp1 = cur.next
            temp2 = cur.next.next.next

            cur.next = cur.next.next

            cur.next.next = temp1

            temp1.next = temp2

            cur = cur.next.next

        return dummy_head.next
