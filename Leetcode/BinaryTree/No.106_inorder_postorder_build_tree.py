class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: Optional[TreeNode]
        """
        # * step1: 递归终止条件
        if not postorder:
            return None
        
        # * step2: 确定根节点
        root = TreeNode(postorder[-1])
        index_root = inorder.index(root.val)
        
        # * step3: 划分左右子树的中序遍历
        left_tree_in = inorder[:index_root]
        right_tree_in = inorder[index_root + 1 :]

        # * step4: 划分左右子树的后序遍历
        left_tree_post = postorder[: len(left_tree_in)]
        right_tree_post = postorder[len(left_tree_in) : len(postorder) - 1]
        
        # * step5: 递归构建左右子树
        root.left = self.buildTree(left_tree_in, left_tree_post)
        root.right = self.buildTree(right_tree_in, right_tree_post)

        return root
