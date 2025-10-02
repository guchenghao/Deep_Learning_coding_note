import collections


class Solution(object):
    def searchBST(self, root, val):
        """
        :type root: Optional[TreeNode]
        :type val: int
        :rtype: Optional[TreeNode]
        """
        # * 很传统的解法，层序遍历 (广度优先遍历)
        if not root:
            return None

        queue_level = collections.deque([root])

        queue_size = 1

        while queue_level:
            node = queue_level.popleft()
            queue_size -= 1
            if node.val == val:
                return node

            if node.left:
                queue_level.append(node.left)

            if node.right:
                queue_level.append(node.right)

            if queue_size == 0:
                queue_size = len(queue_level)

        return None


class Solution(object):
    def searchBST(self, root, val):
        """
        :type root: Optional[TreeNode]
        :type val: int
        :rtype: Optional[TreeNode]
        """
        # * 递归解法 (深度优先遍历)

        if not root:
            return None

        if root.val == val:
            return root

        elif root.val > val:
            result = self.searchBST(root.left, val)

        else:
            result = self.searchBST(root.right, val)

        return result
