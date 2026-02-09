import matplotlib.pyplot as plt
import numpy as np

text_file = "/home/sakurai/Chiba3Party/Trans_copy/chiba_2.txt"
f_list = {}
labels = []

with open(text_file) as t:
    for line in t:
        _, _, filler = line.strip().split(' ')
        if filler in f_list:
            f_list[filler] += 1
        else :
            f_list[filler] = 1
            labels.append(filler)


value = f_list.values()

print(f_list)