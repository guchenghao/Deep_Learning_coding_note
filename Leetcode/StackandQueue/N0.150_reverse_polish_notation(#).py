# * 逆波兰式就是后缀表达式
# * 从这道题，我们可以知道栈这种数据结构擅长做消除操作，括号匹配，相邻相同字符，后缀表达式


class Solution(object):

    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        stack = []

        for token in tokens:

            # print(stack)

            if token == "+":
                num1 = stack.pop()
                num2 = stack.pop()
                stack.append(num2 + num1)

            elif token == "*":
                num1 = stack.pop()
                num2 = stack.pop()
                stack.append(num2 * num1)

            elif token == "-":
                num1 = stack.pop()
                num2 = stack.pop()
                stack.append(num2 - num1)

            elif token == "/":
                num1 = stack.pop()
                num2 = stack.pop()
                stack.append(int(num2 / float(num1)))  # * 注意

            else:
                stack.append(int(token))

        return stack[-1]
