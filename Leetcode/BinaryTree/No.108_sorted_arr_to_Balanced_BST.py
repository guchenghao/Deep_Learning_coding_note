# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: Optional[TreeNode]
        """
        if not nums:
            return None

        mid_root = len(nums) // 2  # * 选择中间节点作为根节点, 考虑到平衡性

        root = TreeNode(nums[mid_root])

        left_tree = nums[:mid_root]
        right_tree = nums[mid_root + 1 :]

        root.left = self.sortedArrayToBST(left_tree)
        root.right = self.sortedArrayToBST(right_tree)

        return root
