class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        if not root: return 0
        self.left_sum_res = 0
        flag = 3
        def getleft_sum(node, flag):
            if not node.left and not node.right and flag == 1:
                self.left_sum_res += node.val

            if node.left:
                flag = 1
                getleft_sum(node.left, flag)
            

            if node.right:
                flag = 2
                getleft_sum(node.right, flag)
        

        getleft_sum(root, flag)

        return self.left_sum_res