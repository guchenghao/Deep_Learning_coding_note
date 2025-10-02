class Solution(object):
    def __init__(self):
        # * 采用双指针的中序遍历方法
        # * 需要将pre指针设置为全局变量
        self.pre = None
        self.result = float("inf")

    def min_node_diff(self, cur):
        # * 左中右遍历
        if not cur:
            return

        self.min_node_diff(cur.left) # * 左
        if self.pre:
            self.result = min(self.result, cur.val - self.pre.val)

        self.pre = cur

        self.min_node_diff(cur.right) # * 右

    def getMinimumDifference(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """

        self.min_node_diff(root)

        return self.result
