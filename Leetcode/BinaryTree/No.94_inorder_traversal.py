# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    # * 递归法
    def inorderTraversal(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[int]
        """
        if not root:
            return []

        self.result = []

        def inorderList(node):

            if node:
                inorderList(node.left)
                self.result.append(node.val)
                inorderList(node.right)

        inorderList(root)

        return self.result
