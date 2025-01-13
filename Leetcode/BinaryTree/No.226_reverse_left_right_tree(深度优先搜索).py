# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def invertTree(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: Optional[TreeNode]
        """
        # * 前序和后序遍历解题较为简单，不要使用中序遍历解题
        if not root:
            return root

        def invert(node):

            if node:
                node.right, node.left = node.left, node.right
                invert(node.left)
                invert(node.right)

        invert(root)

        return root
