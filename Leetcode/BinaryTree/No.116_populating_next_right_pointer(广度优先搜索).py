"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""


class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if not root:
            return
        queue_level = collections.deque([root])

        queue_size = 1

        while queue_level:
            queue_size -= 1
            node = queue_level.popleft()
            if queue_level and queue_size != 0:
                node.next = queue_level[0]

            if node.left:
                queue_level.append(node.left)

            if node.right:
                queue_level.append(node.right)

            if queue_size == 0:
                queue_size = len(queue_level)

        return root
