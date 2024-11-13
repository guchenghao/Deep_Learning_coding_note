#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/Deep_Learning_coding_note/Leetcode/kamacode_No.58_interval_sum.py
# Project: /Users/guchenghao/Deep_Learning_coding_note/Leetcode
# Created Date: Wednesday, November 13th 2024, 5:20:02 pm
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



def main():
    length_arr = int(input())
    
    perfix_sum = [0] * (length_arr + 1)
    acc = 0
    
    for i in range(length_arr):
        acc += int(input())
        perfix_sum[i + 1] = acc
    

    while True:
        try:
            nums = input().split()
        except:
            
            break
        
        print(perfix_sum[int(nums[1]) + 1] - perfix_sum[int(nums[0])])



if __name__ == '__main__':
    main()