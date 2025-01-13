# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[List[int]]
        """
        if not root:
            return []
        
        # * 利用队列来实现广度优先搜索
        queue_level = collections.deque([root])

        result = []
        temp = []

        queue_size = 1

        while queue_level:
            queue_size -= 1
            node = queue_level.popleft()
            temp.append(node.val)

            if node.left:
                queue_level.append(node.left)

            if node.right:
                queue_level.append(node.right)

            if queue_size == 0:
                result.append(temp)
                temp = []
                queue_size = len(queue_level)

        return result
