# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[str]
        """
        if not root:
            return []

        result = []
        path = ""
        
        # * 深度优先搜索，前序遍历，因为要保存之前的路径
        def getPath(node, path, result):

            if not node.left and not node.right:
                result.append(path + str(node.val))

            if node.left:
                getPath(node.left, path + str(node.val) + "->", result)

            if node.right:
                getPath(node.right, path + str(node.val) + "->", result)

        getPath(root, path, result)

        return result
