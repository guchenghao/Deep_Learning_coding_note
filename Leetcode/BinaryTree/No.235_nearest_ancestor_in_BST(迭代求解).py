import collections


class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        # * 利用BST的特性 （左子树 < 根 < 右子树）
        # * 采用层序遍历的方式可以直接找到最近公共祖先
        if not root:
            return None
        queue_level = collections.deque([root])

        queue_size = 1

        while queue_level:
            node = queue_level.popleft()

            queue_size -= 1

            if node.val > p.val and node.val > q.val:
                if node.left:
                    queue_level.append(node.left)

            elif node.val < p.val and node.val < q.val:
                if node.right:

                    queue_level.append(node.right)

            else:
                return node

            if queue_size == 0:
                queue_size = len(queue_level)

        return None
