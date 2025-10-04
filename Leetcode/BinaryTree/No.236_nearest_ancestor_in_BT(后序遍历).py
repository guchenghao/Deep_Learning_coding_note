class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        # * 后序遍历
        # * 从下往上找最近公共祖先
        # * 先找到P和Q节点
        if not root:
            return None
        if root == p or root == q:
            return root

        left_tree = self.lowestCommonAncestor(root.left, p, q)
        right_tree = self.lowestCommonAncestor(root.right, p, q)
        
        
        # * 分情况讨论
        # * 1. 左右子树都找到了P或Q节点，说明当前节点就是最近公共祖先
        # * 2. 左右子树都没有找到P或Q节点，说明当前节点不是最近公共祖先，返回None
        # * 3. 左子树没有找到P或Q节点，右子树找到了P或Q节点，说明当前节点不是最近公共祖先，返回右子树找到的节点
        # * 4. 右子树没有找到P或Q节点，左子树找到了P或Q节点，说明当前节点不是最近公共祖先，返回左子树找到的节点
        if left_tree and right_tree:
            return root

        if not left_tree and not right_tree:
            return None

        if not left_tree and right_tree:
            return right_tree

        if left_tree and not right_tree:
            return left_tree
