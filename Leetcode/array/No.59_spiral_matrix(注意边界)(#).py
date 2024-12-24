#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/Deep_Learning_coding_note/Leetcode/No.59_rotation_matrix.py
# Project: /Users/guchenghao/Deep_Learning_coding_note/Leetcode
# Created Date: Tuesday, November 12th 2024, 8:50:11 pm
# Author: GU CHENGHAO
# -----
# 2024
# Last Modified: GU CHENGHAO
# Modified By: GU CHENGHAO
# -----
# Copyright (c) 2024 Personal File
# 
# MIT License
# 
# Copyright (c) 2024 GU CHENGHAO
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----
# HISTORY: 
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###

class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        spiral_matrix = [[0 for _ in range(n)] for _ in range(n)]
        startx = 0
        starty = 0
        offset = 1  # * 记录圈数
        num = 1

        # * 这道题的关键之处在于循环不变量，左闭右开的区间，这样处理边界条件才不会乱
        for offset in range(1, n // 2 + 1):
            # * left -> right
            for i in range(starty, n - offset):
                spiral_matrix[startx][i] = num
                num += 1
            
            # * up -> bottom
            for i in range(startx, n - offset):
                spiral_matrix[i][n - offset] = num
                num += 1

            # * right -> left
            for i in range(n - offset, starty, -1):
                spiral_matrix[n - offset][i] = num
                num += 1
            
            # * bottom -> up
            for i in range(n - offset, startx, -1):
                spiral_matrix[i][starty] = num
                num += 1
            
            startx += 1
            starty += 1
        
        if n % 2 != 0:
            spiral_matrix[n // 2][n // 2] = num
        
        return spiral_matrix

