class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: Optional[ListNode]
        :type val: int
        :rtype: Optional[ListNode]
        """
        # * 处理头节点
        while head and head.val == val:
            head = head.next

        
        cur = head  # * 设置一个遍历指针
        while cur and cur.next:
            if cur.next.val == val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        


        return head




# * 虚拟头节点
class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: Optional[ListNode]
        :type val: int
        :rtype: Optional[ListNode]
        """


        dummy_head = ListNode(val=0, next=head)


        cur = dummy_head  # * 设置一个遍历指针
        while cur.next:

            if cur.next.val == val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        

        return dummy_head.next