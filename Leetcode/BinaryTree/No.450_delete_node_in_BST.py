class Solution(object):
    def deleteNode(self, root, key):
        """
        :type root: Optional[TreeNode]
        :type key: int
        :rtype: Optional[TreeNode]
        """
        # * 递归版
        # * 删除节点的操作是通过返回值来实现的
        if not root:
            return None

        if root.val == key:
            if not root.left and not root.right:
                return None
            elif root.left and not root.right:
                return root.left

            elif not root.left and root.right:
                return root.right

            else:
                cur = root.right

                while cur.left is not None:
                    cur = cur.left

                cur.left = root.left

                return root.right

        if root.val > key:
            root.left = self.deleteNode(root.left, key) # * 通过返回值来实现删除节点的操作

        if root.val < key:
            root.right = self.deleteNode(root.right, key) # * 通过返回值来实现删除节点的操作

        return root


class Solution:
    def deleteOneNode(self, target: TreeNode) -> TreeNode:
        """
        将目标节点（删除节点）的左子树放到目标节点的右子树的最左面节点的左孩子位置上
        并返回目标节点右孩子为新的根节点
        是动画里模拟的过程
        """
        # * deleteOneNode 函数用于删除一个节点，并返回新的子树根节点
        if target is None:
            return target
        if target.right is None:
            return target.left
        cur = target.right
        while cur.left:
            cur = cur.left
        cur.left = target.left
        return target.right

    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if root is None:
            return root
        cur = root
        pre = None  # 记录cur的父节点，用来删除cur
        while cur:
            if cur.val == key:
                break
            pre = cur
            if cur.val > key:
                cur = cur.left
            else:
                cur = cur.right
        if pre is None:  # 如果搜索树只有头结点
            return self.deleteOneNode(cur)
        # pre 要知道是删左孩子还是右孩子
        if pre.left and pre.left.val == key:
            pre.left = self.deleteOneNode(cur)
        if pre.right and pre.right.val == key:
            pre.right = self.deleteOneNode(cur)
        return root
