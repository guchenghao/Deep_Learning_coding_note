# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: bool
        """
        # * 广度优先搜索
        if not root:
            return True

        # * 利用队列来实现
        bfs_left = [root.left]
        bfs_right = [root.right]

        while bfs_left and bfs_right:

            node_left = bfs_left.pop(0)
            node_right = bfs_right.pop(0)

            if not node_left and not node_right:
                continue

            if (not node_left and node_right) or (node_left and not node_right):
                return False

            if node_left.val != node_right.val:
                return False

            bfs_left.append(node_left.left)
            bfs_left.append(node_left.right)
            # * 因为是判断左右子树是否对称，因此左右子树的队列，节点入队的顺序应该是相反的
            bfs_right.append(node_right.right)
            bfs_right.append(node_right.left)

        return True




class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: bool
        """
        # * 深度优先搜索且后序遍历
        # * compare的过程，实际上是从上往下比较，如果上面一层，比较的结果有问题，就不继续往下递归了，就往上return比较结果
        if not root:
            return True

        def compare(left, right):
            if not left and not right:
                return True

            elif not left and right:
                return False

            elif not right and left:
                return False

            elif left.val != right.val:
                return False

            else:

                outside = compare(left.left, right.right)
                inside = compare(left.right, right.left)

                result = outside & inside

                return result

        return compare(root.left, root.right)
