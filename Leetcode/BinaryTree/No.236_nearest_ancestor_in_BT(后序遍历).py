class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        # * 后序遍历
        if not root:
            return None
        if root == p or root == q:
            return root

        left_tree = self.lowestCommonAncestor(root.left, p, q)
        right_tree = self.lowestCommonAncestor(root.right, p, q)

        if left_tree and right_tree:
            return root

        if not left_tree and not right_tree:
            return None

        if not left_tree and right_tree:
            return right_tree

        if left_tree and not right_tree:
            return left_tree
