#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/Deep_Learning_coding_note/Leetcode/No.209_smallest_interval.py
# Project: /Users/guchenghao/Deep_Learning_coding_note/Leetcode
# Created Date: Tuesday, November 12th 2024, 7:23:37 pm
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

# * 暴力求解（超时）
class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        smallest_interval = None

        for i in range(len(nums)):
            right = i
            acc = 0
            count = 0
            
            while right <= len(nums) - 1:
                acc += nums[right]
                count += 1
                right += 1
                if acc >= target:

                    if not smallest_interval:
                        smallest_interval = count
                    else:
                        smallest_interval = min(smallest_interval, count)
                    
                    break
                
                
        if not smallest_interval:
            return 0
        else:
            return smallest_interval



class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        # * 滑动窗口法：要点在于滑动窗口的终止位置j正常遍历即可，起始位置start需要动态调整
        start = 0
        acc = 0
        smallest_interval = None

        for j in range(len(nums)):
            acc += nums[j]
            
            if acc >= target:
                if not smallest_interval:
                    smallest_interval = j + 1 - start
                
                while acc >= target and start <= j:
                    smallest_interval = min(smallest_interval, j + 1 - start)
                    acc -= nums[start]
                    start += 1
        

        if not smallest_interval:
            return 0
        else:
            return smallest_interval