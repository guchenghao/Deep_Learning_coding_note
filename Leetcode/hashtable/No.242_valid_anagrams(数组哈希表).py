
# * 采用数组作为hash表形式更加简洁
# * 因为题目中字符串组成字母均为小写字母，没有大写字母，因此利用数组来作为哈希表最为简洁
class Solution(object):

    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """

        hash_arr = [0] * 26

        for char in s:
            hash_arr[ord(char) - ord("a")] += 1

        for char in t:
            hash_arr[ord(char) - ord("a")] -= 1

        if all(count == 0 for count in hash_arr):
            return True

        else:
            return False



# * 字典（map）的形式的hash表
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        alphabet_dict = {}

        for char in s:
            if char in alphabet_dict:
                alphabet_dict[char] += 1
            else:
                alphabet_dict[char] = 1

        for char in t:
            if char in alphabet_dict:
                alphabet_dict[char] -= 1

            else:
                return False

        if all(value == 0 for value in alphabet_dict.values()):
            return True

        else:
            return False
