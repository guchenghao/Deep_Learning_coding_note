#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/Deep_Learning_coding_note/Leetcode/kamacode_No.44_divide_land.py
# Project: /Users/guchenghao/Deep_Learning_coding_note/Leetcode
# Created Date: Wednesday, November 13th 2024, 7:11:01 pm
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



# * 计算行和和列和
# * 总土地价值不变
def main():
    data = input().split()
    
    n = int(data[0])
    m = int(data[1])
    
    value_matrix = []
    horizon_value = [0] * n
    vertical_value = [0] * m
    total_sum = 0 # * 总和，很重要
    minimum_diff = float("inf")
    
    # * 横向
    for i in range(n):
        horizon = input().split()
        value_matrix.append(horizon)
        for value in horizon:
            horizon_value[i] += int(value)
        total_sum += horizon_value[i]
    

    # * 纵向
    for i in range(m):
        for j in range(n):
            vertical_value[i] += int(value_matrix[j][i])
        
    # * 横切
    horizon_cut = 0
    for i in range(n - 1):
        horizon_cut += horizon_value[i]
        
        minimum_diff = min(minimum_diff, abs(total_sum - 2 * horizon_cut))

    
    
    # * 纵切
    vertical_cut = 0
    for i in range(m - 1):
        vertical_cut += vertical_value[i]
        
        minimum_diff = min(minimum_diff, abs(total_sum - 2 * vertical_cut))

    
        
    print(minimum_diff)
   

 
if __name__ == '__main__':
    main()