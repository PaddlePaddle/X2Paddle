import sys
import math
val1 = map(float, open(sys.argv[1]).read().strip().split('\n'))
val2 = map(float, open(sys.argv[2]).read().strip().split('\n'))

if len(val1) != len(val2):
    raise Exception("Not Same Length")

max_diff = 0
avg_diff = 0
for i in range(len(val1)):
    diff = math.fabs(val1[i] - val2[i])
    if diff > max_diff:
        max_diff = diff
    avg_diff += diff
avg_diff /= len(val1)
print("max_diff: {}\tavg_diff: {}".format(max_diff, avg_diff))
