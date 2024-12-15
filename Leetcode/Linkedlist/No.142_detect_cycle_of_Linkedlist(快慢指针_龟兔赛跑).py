class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return None

        fast = head
        slow = head

        # * 利用快慢指针来判断是否有环
        # * 快指针走2步，慢指针走1步
        while True:
            if fast and fast.next:

                fast = fast.next.next
                slow = slow.next

                if fast == slow:
                    break

            else:
                return None

        # * 如果有环，从起始点和相遇点同时出发，第一次相遇的节点就是环的入口节点
        index1 = head
        index2 = fast

        while True:
            if index1 == index2:
                break
            index1 = index1.next
            index2 = index2.next

        return index1
