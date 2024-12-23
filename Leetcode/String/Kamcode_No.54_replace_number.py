def main():
    str_input = list(input())

    for i in range(len(str_input)):
        if ord(str_input[i]) >= ord("0") and ord(str_input[i]) <= ord("9"):
            str_input[i] = "number"

    print("".join(str_input))


if __name__ == "__main__":
    main()
