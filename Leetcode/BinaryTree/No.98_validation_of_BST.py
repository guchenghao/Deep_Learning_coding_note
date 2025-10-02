class Solution(object):
    def __init__(self):
        self.inorder_arr = []

    def inorderBST(self, node):
        # * 对于BST来说，中序遍历的结果是一个升序数组
        # * 这是一个非常重要的性质
        if not node:
            return

        self.inorderBST(node.left)
        self.inorder_arr.append(node.val)
        self.inorderBST(node.right)

    def isValidBST(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: bool
        """
        if not root:
            return False
        if not root.left and not root.right:
            return True
        self.inorder_arr = []

        self.inorderBST(root)
        # print(self.inorder_arr)

        i = 0
        j = 1

        while i < len(self.inorder_arr) - 1:
            if self.inorder_arr[i] < self.inorder_arr[j]:
                i += 1
                j += 1
                continue
            else:
                return False

        return True
