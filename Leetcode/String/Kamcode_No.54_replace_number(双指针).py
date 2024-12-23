
# * 解法1
def main():
    str_input = list(input())

    for i in range(len(str_input)):
        if ord(str_input[i]) >= ord("0") and ord(str_input[i]) <= ord("9"):
            str_input[i] = "number"

    print("".join(str_input))


if __name__ == "__main__":
    main()





# * 解法2
# * 空间复杂度O(1)的解法
def main():
    str_input = list(input())
    count = 0
    left = len(str_input) - 1
    start = left + 1

    for i in range(len(str_input)):
        if ord(str_input[i]) >= ord("0") and ord(str_input[i]) <= ord("9"):
            count += 1

    str_input += [0] * (count * 5)

    right = len(str_input) - 1

    while left >= 0:
        if ord(str_input[left]) >= ord("0") and ord(str_input[left]) <= ord("9"):
            str_input[right] = "r"
            right -= 1
            str_input[right] = "e"
            right -= 1
            str_input[right] = "b"
            right -= 1
            str_input[right] = "m"
            right -= 1
            str_input[right] = "u"
            right -= 1
            str_input[right] = "n"
            right -= 1

        else:
            str_input[right] = str_input[left]
            right -= 1

        left -= 1

    print("".join(str_input))


if __name__ == "__main__":
    main()
