# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def minDepth(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        if not root:
            return 0
        queue_level = collections.deque([root])

        queue_size = 1
        level = 1

        while queue_level:
            queue_size -= 1
            node = queue_level.popleft()

            if not node.left and not node.right:
                return level

            if node.left:
                queue_level.append(node.left)

            if node.right:
                queue_level.append(node.right)

            if queue_size == 0:
                level += 1
                queue_size = len(queue_level)

        return level
