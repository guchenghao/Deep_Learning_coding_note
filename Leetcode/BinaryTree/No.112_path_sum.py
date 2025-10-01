class Solution(object):
    def hasPathSum(self, root, targetSum):
        """
        :type root: Optional[TreeNode]
        :type targetSum: int
        :rtype: bool
        """
        if not root:
            return False

        path_sum = root.val

        self.result = False

        def path_sum_traversal(node, path_sum):
            if not node.left and not node.right:
                if path_sum == targetSum:
                    self.result = True

            if node.left:
                path_sum_traversal(node.left, path_sum + node.left.val)

            if node.right:
                path_sum_traversal(node.right, path_sum + node.right.val)

        path_sum_traversal(root, path_sum)

        return self.result
