
# * 用Python数组的函数偷鸡（图一乐）
class MyLinkedList:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.linklist = []
        

    def get(self, index):
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        :type index: int
        :rtype: int
        """
        if index >= len(self.linklist) or index < 0:
            return -1
        else:
            return self.linklist[index]
        

    def addAtHead(self, val):
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        :type val: int
        :rtype: void
        """
        self.linklist.insert(0, val)

    def addAtTail(self, val):
        """
        Append a node of value val to the last element of the linked list.
        :type val: int
        :rtype: void
        """
        self.linklist.append(val)

    def addAtIndex(self, index, val):
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        :type index: int
        :type val: int
        :rtype: void
        """
        if index == len(self.linklist):
            self.linklist.append(val)
        elif index > len(self.linklist):
            return 
        else:
            self.linklist.insert(index, val)
        

    def deleteAtIndex(self, index):
        """
        Delete the index-th node in the linked list, if the index is valid.
        :type index: int
        :rtype: void
        """
        if index >= len(self.linklist) or index < 0:
            return
        else:
            self.linklist.pop(index)






class LinkNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next



class MyLinkedList(object):

    def __init__(self):
        self.dummy_head = LinkNode()  # * 设置一个虚拟头指针
        self.size = 0
        

    def get(self, index):
        """
        :type index: int
        :rtype: int
        """
        if index >= self.size or index < 0:
            return -1
        
        cur = self.dummy_head
        for i in range(index):
            cur = cur.next
        
        return cur.next.val


    def addAtHead(self, val):
        """
        :type val: int
        :rtype: None
        """
        self.dummy_head.next = LinkNode(val, self.dummy_head.next)
        self.size += 1

    def addAtTail(self, val):
        """
        :type val: int
        :rtype: None
        """
        cur = self.dummy_head

        while cur.next:
            cur = cur.next
        
        cur.next = LinkNode(val)
        self.size +=1

    def addAtIndex(self, index, val):
        """
        :type index: int
        :type val: int
        :rtype: None
        """
        if index > self.size or index < 0:
            return
        

        cur = self.dummy_head
        for i in range(index):
            cur = cur.next
        
        cur.next = LinkNode(val, cur.next)
        self.size += 1


    def deleteAtIndex(self, index):
        """
        :type index: int
        :rtype: None
        """
        if index < 0 or index >= self.size:
            return

        cur = self.dummy_head
        for i in range(index):
            cur = cur.next
        
        cur.next = cur.next.next
        self.size -= 1


# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)

