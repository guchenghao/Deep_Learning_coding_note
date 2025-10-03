import collections


class Solution(object):
    def insertIntoBST(self, root, val):
        """
        :type root: Optional[TreeNode]
        :type val: int
        :rtype: Optional[TreeNode]
        """
        # * 迭代版
        # * 利用队列实现层序遍历
        if not root:
            return TreeNode(val)
        queue_level = collections.deque([root])

        queue_size = 1

        while queue_level:
            node = queue_level.popleft()

            queue_size -= 1

            if node.val > val:
                if node.left:
                    queue_level.append(node.left)

                else:
                    node.left = TreeNode(val)
                    break

            if node.val < val:
                if node.right:
                    queue_level.append(node.right)

                else:
                    node.right = TreeNode(val)
                    break

            if queue_size == 0:
                queue_size = len(queue_level)

        return root
