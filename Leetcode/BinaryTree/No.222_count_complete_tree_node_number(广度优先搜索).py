# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def countNodes(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        if not root:
            return 0

        self.count = 1

        # * 利用队列来实现广度优先搜索
        queue_level = collections.deque([root])

        queue_size = 1

        while queue_level:
            queue_size -= 1
            node = queue_level.popleft()

            if node.left:
                queue_level.append(node.left)

            if node.right:
                queue_level.append(node.right)

            if queue_size == 0:
                self.count += len(queue_level)
                queue_size = len(queue_level)

        return self.count
