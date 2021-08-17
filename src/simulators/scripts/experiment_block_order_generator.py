# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import random

a = ['n', 'f', 'c']
inds = [0,1]

final_list = []

index_first = [random.randint(0,1) for i in range(3)]
index_second = [int(not bool(i) )for i in index_first]
index = []
index.extend(index_first)
index.extend(index_second)

final_list.extend(a)
final_list.extend(a)

order = [ s + '_' + str(ind) for ind, s in zip(index, final_list)]
first_part = order[:3]
second_part = order[3:]
random.shuffle(first_part)
random.shuffle(second_part)
final_order = []
final_order.extend(first_part)
final_order.extend(second_part)
print(final_order)
