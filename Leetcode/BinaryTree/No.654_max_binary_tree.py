class Solution(object):
    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: Optional[TreeNode]
        """
        # * 这道题和No.106类似，步骤很相似
        # * 只不过No.106是根据中序和后序遍历结果构造二叉树，需要对中序和后序数组都进行划分
        if not nums:
            return None

        if len(nums) == 1:
            return TreeNode(nums[0])

        max_value = max(nums)
        max_index = nums.index(max_value)
        root = TreeNode(max_value)

        left_tree = nums[:max_index]
        right_tree = nums[max_index + 1 :]

        root.left = self.constructMaximumBinaryTree(left_tree)
        root.right = self.constructMaximumBinaryTree(right_tree)

        return root
