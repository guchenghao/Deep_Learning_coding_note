# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[int]
        """
        if not root:
            return []

        self.result = []

        self.preorderList(root)

        return self.result

    def preorderList(self, node):

        if node:
            self.result.append(node.val)
            self.preorderList(node.left)
            self.preorderList(node.right)
