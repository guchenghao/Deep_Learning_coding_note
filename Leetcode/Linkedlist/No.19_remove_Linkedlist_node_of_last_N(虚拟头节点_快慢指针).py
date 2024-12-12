class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: Optional[ListNode]
        :type n: int
        :rtype: Optional[ListNode]
        """
        dummy_head = ListNode(next=head)
        
        # * 采用快慢指针的方式能够方便地找到链表中的“倒数”第N个节点的位置
        # * 如果不是倒数的话，这道题就不用快慢指针，直接让cur指针指向删除节点的前一个节点即可

        fast = dummy_head
        slow = dummy_head

        for _ in range(n + 1):   # * 这里设置n + 1是为了让slow指针指向删除节点的前一个节点
            fast = fast.next

        while fast:
            fast = fast.next
            slow = slow.next

        temp = slow.next.next

        slow.next = temp

        return dummy_head.next
