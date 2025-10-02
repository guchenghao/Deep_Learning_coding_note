class Solution(object):
    def mergeTrees(self, root1, root2):
        """
        :type root1: Optional[TreeNode]
        :type root2: Optional[TreeNode]
        :rtype: Optional[TreeNode]
        """
        if not root1 and root2:
            return root2

        if not root2 and root1:
            return root1

        if not root1 and not root2:
            return None

        root = TreeNode(root1.val + root2.val)

        root.left = self.mergeTrees(root1.left, root2.left)
        root.right = self.mergeTrees(root1.right, root2.right)

        return root
