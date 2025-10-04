class Solution(object):
    def trimBST(self, root, low, high):
        """
        :type root: Optional[TreeNode]
        :type low: int
        :type high: int
        :rtype: Optional[TreeNode]
        """
        # * 这道题也采用了返回值的方式来实现对节点的删除
        # * 充分利用了BST的特性 （左子树 < 根 < 右子树）
        if not root:
            return None

        if root.val < low:
            right = self.trimBST(root.right, low, high)
            return right

        elif root.val > high:
            left = self.trimBST(root.left, low, high)
            return left

        else:
            root.left = self.trimBST(root.left, low, high)

            root.right = self.trimBST(root.right, low, high)

        return root
