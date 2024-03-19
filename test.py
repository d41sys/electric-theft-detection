from typing import List

intervals = [[1,2],[4,5],[6,7],[8,10],[12,16]]
newInterval = [3,8]
res = []

for i, [start, end] in enumerate(intervals):
    if end < newInterval[0]:
        res.append([start, end])
    else: # newInterval[0] <= end
        if start > newInterval[1]:
            res.append(newInterval)
            res.append([start, end])
        else:
            newInterval[0] = min(start, newInterval[0])
            newInterval[1] = max(end, newInterval[1])
print(res)