import collections
class Solution(object):
    def findBottomLeftValue(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        queue_level = collections.deque([root])
        queue_size = 1
        leftest_arr = []
        while queue_level:
            node = queue_level.popleft()
            queue_size -= 1
            if not node.left and not node.right:
                leftest_arr.append(node)
            if node.left:
                queue_level.append(node.left)

            if node.right:
                queue_level.append(node.right)

            if queue_size == 0:
                leftest_node = leftest_arr[0] if leftest_arr else None
                queue_size = len(queue_level)
                leftest_arr = []

        return leftest_node.val


class Solution(object):
    # * 深度优先搜索(左子树优先)
    def findBottomLeftValue(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        self.max_depth = float("-inf")
        self.result = None
        self.traversal(root, 0)
        return self.result

    def traversal(self, node, depth):
        if not node.left and not node.right:
            if depth > self.max_depth:
                self.max_depth = depth
                self.result = node.val
            return

        if node.left:
            # * depth + 1的过程就是回溯的过程
            self.traversal(node.left, depth + 1)
        if node.right:
            self.traversal(node.right, depth + 1)
