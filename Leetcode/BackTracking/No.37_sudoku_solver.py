class Solution(object):
    # * python超时
    def is_valid(self, row, col, num, board):
        for i in range(len(board)):
            if board[i][col] == num:
                return False

        for j in range(len(board)):
            if board[row][j] == num:
                return False

        startcol = col // 3 * 3
        startrow = row // 3 * 3

        for i in range(startrow, startrow + 3):
            for j in range(startcol, startcol + 3):
                if board[i][j] == num:
                    return False

        return True

    def backtracking(self, board):
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] != ".":
                    continue
                for k in range(1, 10):
                    if self.is_valid(i, j, str(k), board):
                        board[i][j] = str(k)
                        if self.backtracking(board):
                            return True
                        board[i][j] = "."
                return False

        return True

    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """

        return self.backtracking(board)


class Solution(object):
    # * python优化版本
    def is_valid(self, row, col, num, board):
        for i in range(len(board)):
            if board[i][col] == num or board[row][i] == num:
                return False

            if board[(row // 3) * 3 + i // 3][(col // 3) * 3 + i % 3] == num:
                return False

        return True

    def backtracking(self, board):
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] != ".":
                    continue
                for k in range(1, 10):
                    if self.is_valid(i, j, str(k), board):
                        board[i][j] = str(k)
                        if self.backtracking(board):
                            return True
                        board[i][j] = "."
                return False

        return True

    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """

        return self.backtracking(board)
