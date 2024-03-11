from typing import *

def minimumLength(s: str) -> int:
    if len(s) == 1:
        return 1
    length = len(s)
    
    temp = 's'
    left = 0
    right = len(s) - 1 
    while True:
        if s[left] == s[right] and left <= right:
            temp = s[left]
            left += 1
            right -= 1
            continue
        
        if s[left] == temp:
            left += 1
            continue
        
        if s[right] == temp:
            right -= 1
            continue

        if s[left] != s[right] or left > right:
            break
        

    return right - left + 1

minimumLength("abbbbbbbbbbbbbbbbbbba")