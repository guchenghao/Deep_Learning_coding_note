class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        magazine_dict = {}

        for char in magazine:
            if char in magazine_dict:
                magazine_dict[char] += 1
            else:
                magazine_dict[char] = 1

        for char in ransomNote:
            if char in magazine_dict:
                if magazine_dict[char] <= 0:
                    return False

                magazine_dict[char] -= 1

            else:
                return False

        return True
