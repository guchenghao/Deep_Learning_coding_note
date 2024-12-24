mode_str = "aabaaf"
target_str = "aabaabaafaabaaf"
next = [0] * len(mode_str)


# * 获得前缀数组
def getPrefix_arr(next, mode_str):
    j = 0   # * 前缀末尾
    next[0] = 0
    
    # * i 表示后缀末尾
    for i in range(1, len(mode_str)):
        while j > 0 and mode_str[i] != mode_str[j]:
            j = next[j - 1]
            
        
        if mode_str[i] == mode_str[j]:
            j += 1
        
        
        next[i] = j
    
    
    return next

next = getPrefix_arr(next, mode_str)


print(next)

j = 0

for i in range(len(target_str)):
    # * 如果不匹配的话, j进行回退
    while j > 0 and mode_str[j] != target_str[i]:
         j = next[j - 1]
        
    
    if mode_str[j] == target_str[i]:  # * 如果匹配，i++和j++
        j += 1
    
    
    if j == len(mode_str):
        print("true")
        break
