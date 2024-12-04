# * 双指针解法
class Solution(object):
    def reverseList(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        prev = None
        cur = head

        while cur:
            # * 需要构建一个临时指针
            temp = cur.next
            cur.next = prev

            prev = cur
            cur = temp

        return prev


# * 递归写法
class Solution(object):
    def reverseList(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: Optional[ListNode]
        """

        def reverse(cur, prev):

            if not cur:
                return prev

            temp = cur.next
            cur.next = prev

            return reverse(temp, cur)

        return reverse(head, None)
