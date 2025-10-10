class Solution(object):
    def __init__(self):
        # * 二维数组的回溯
        self.result = []

    def is_valid(self, row, col, chessboard):
        
        # * 只需要向上检查
        # * 检查列
        for j in range(row):
            if chessboard[j][col] == "Q":
                return False
            
        # * 检查左上对角线和右上对角线
        i = row - 1
        j = col - 1

        while i >= 0 and j >= 0:
            if chessboard[i][j] == "Q":
                return False

            i -= 1
            j -= 1

        i = row - 1
        j = col + 1

        while i >= 0 and j <= len(chessboard) - 1:
            if chessboard[i][j] == "Q":
                return False

            i -= 1
            j += 1

        return True

    def backtracking(self, n, chessboard, row):
        if row == n:
            temp = []
            for i in range(n):
                temp.append("".join(chessboard[i]))
            self.result.append(temp[:])
            return

        for i in range(n):
            # * 每一行只能放一个皇后，行表示递归树的深度，列表式表示递归树的宽度
            if self.is_valid(row, i, chessboard):
                chessboard[row][i] = "Q"
                self.backtracking(n, chessboard, row + 1)
                chessboard[row][i] = "."

    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        chessboard = [["." for _ in range(n)] for _ in range(n)]

        self.backtracking(n, chessboard, 0)

        return self.result
