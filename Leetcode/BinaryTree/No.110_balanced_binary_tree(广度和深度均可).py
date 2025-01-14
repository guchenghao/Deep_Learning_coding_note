# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def getDepth(self, root):
        if not root:
            return 1

        queue_level = collections.deque([root])

        depth = 1

        queue_size = 1

        while queue_level:
            queue_size -= 1
            node = queue_level.popleft()

            if node.left:
                queue_level.append(node.left)

            if node.right:
                queue_level.append(node.right)

            if queue_size == 0:
                depth += 1
                queue_size = len(queue_level)

        return depth

    def isBalanced(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: bool
        """
        # * 广度优先搜索
        if not root:
            return True

        queue_level = collections.deque([root])

        while queue_level:
            node = queue_level.popleft()
            if abs(self.getDepth(node.left) - self.getDepth(node.right)) > 1:
                return False

            if node.left:
                queue_level.append(node.left)

            if node.right:
                queue_level.append(node.right)

        return True



class Solution(object):

    def isBalanced(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: bool
        """
        # * 深度优先搜索，后序遍历
        # * 这种需要收集左子树或右子树的信息的情况下，最好使用后序遍历
        if not root:
            return True

        def getheight(node):
            if not node:
                return 0

            left_height = getheight(node.left)
            if left_height == -1:
                return -1

            right_height = getheight(node.right)
            if right_height == -1:
                return -1

            if abs(left_height - right_height) > 1:
                result = -1

            else:
                result = 1 + max(left_height, right_height)

            return result

        final_height = getheight(root)

        return True if final_height != -1 else False
