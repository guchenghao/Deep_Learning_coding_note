# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[float]
        """
        if not root:
            return []
        queue_level = collections.deque([root])

        result = []
        temp = 0
        count = 0

        queue_size = 1

        while queue_level:
            queue_size -= 1
            count += 1
            node = queue_level.popleft()
            temp += node.val

            if node.left:
                queue_level.append(node.left)

            if node.right:
                queue_level.append(node.right)

            if queue_size == 0:
                result.append(int(temp) / float(count))
                count = 0
                temp = 0
                queue_size = len(queue_level)

        return result
