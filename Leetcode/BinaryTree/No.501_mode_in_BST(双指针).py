class Solution(object):
    def __init__(self):
        self.result = []
        self.max_count = 0
        self.cur_count = 1
        self.pre = None

    def inorder_traversal(self, cur):
        # * 左中右遍历 (中序遍历)
        if not cur:
            return

        self.inorder_traversal(cur.left)
        
        # * 处理当前节点并计算count
        if not self.pre:
            self.cur_count = 1
        elif self.pre.val == cur.val:
            self.cur_count += 1
        else:
            self.cur_count = 1

        self.pre = cur
        
        # * 再来处理max_count和result
        if self.cur_count == self.max_count:
            self.result.append(cur.val)

        if self.cur_count > self.max_count:
            self.max_count = self.cur_count
            self.result = [cur.val]

        self.inorder_traversal(cur.right)

    def findMode(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[int]
        """

        self.inorder_traversal(root)

        return self.result
