"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""


class Solution(object):
    def levelOrder(self, root):
        """
        :type root: Node
        :rtype: List[List[int]]
        """
        if not root:
            return []
        queue_level = collections.deque([root])

        result = []
        temp = []

        queue_size = 1

        while queue_level:
            queue_size -= 1
            node = queue_level.popleft()
            temp.append(node.val)

            if node.children:
                queue_level.extend(node.children)

            if queue_size == 0:
                result.append(temp)
                temp = []
                queue_size = len(queue_level)

        return result
