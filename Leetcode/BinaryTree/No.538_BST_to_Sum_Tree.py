class Solution(object):
    def __init__(self):
        self.result = None

    def tree_sum(self, node):
        # * 反向中序遍历
        # * 先遍历右子树，再遍历根节点，最后遍历左子树
        if not node:
            return

        self.tree_sum(node.right)
        if not self.result:
            self.result = node.val
            node.val = self.result
        else:
            self.result += node.val
            node.val = self.result

        self.tree_sum(node.left)

    def convertBST(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: Optional[TreeNode]
        """

        self.tree_sum(root)

        return root


class Solution(object):
    def __init__(self):
        self.pre = None

    def tree_sum(self, cur):
        # * 反向中序遍历
        # * 双指针法
        if not cur:
            return

        self.tree_sum(cur.right)
        if self.pre:
            cur.val += self.pre.val

        self.pre = cur

        self.tree_sum(cur.left)

    def convertBST(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: Optional[TreeNode]
        """

        self.tree_sum(root)

        return root
