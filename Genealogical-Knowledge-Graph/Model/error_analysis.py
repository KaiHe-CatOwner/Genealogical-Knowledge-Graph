import numpy as np
from collections import Counter
import os

def get_list(data, need_lines, task_list, need_line_list):

    task12 = []
    task3 = []
    task4 = []
    task5 = []
    for index, value in enumerate(data):

        if 12 in task_list:
            if index % need_lines == need_line_list[0]:
                items = eval(value.replace("each_PRF = ", "").replace("\n", ""))
                task12.append(items)
        if 3 in task_list:
            if index % need_lines == need_line_list[1]:
                items = eval(value.replace("each_PRF = ", "").replace("\n", ""))
                task3.append(items)
        if 4 in task_list:
            if index % need_lines == need_line_list[2]:
                items = eval(value.replace("each_PRF = ", "").replace("\n", ""))
                task4.append(items)
        if 5 in task_list:
            if index % need_lines == need_line_list[3]:
                items = eval(value.replace("each_PRF = ", "").replace("\n", ""))
                task5.append(items)

    return task12,task3,task4,task5


def merge_dic(need_list):
    tota_dic = {}

    total_P = {}
    total_R = {}
    total_F = {}

    temp_count = {}

    for i in need_list:
        temp_P = {}
        temp_R = {}
        temp_F = {}
        for key, value in i.items():
            temp_P[key] = value[0]
            temp_R[key] = value[1]
            temp_F[key] = value[2]

        for key, item in temp_P.items():
            if key in temp_count.keys():
                temp_count[key] = temp_count[key] +1
            else:
                temp_count[key] = 1

        total_P = dict(Counter(total_P) + Counter(temp_P))
        total_R = dict(Counter(total_R) + Counter(temp_R))
        total_F = dict(Counter(total_F) + Counter(temp_F))

    for key, values in total_P.items():
        tota_dic[key]=(total_P[key] / temp_count[key], total_R[key]  / temp_count[key], total_F[key] / temp_count[key])

    return sorted(tota_dic.items(), key=lambda item:item[1])



if __name__ == "__main__":

    dir = "../result/detail_performance/Bert/"
    file_list = ["performance_['--Data_ID=0', '--GPU=0', '--Task_list=[12,3,4,5]', '--Loss_weight_list=[0.6, 0.2, 0.2, 0.5]'].txt",
                 "performance_['--Data_ID=1', '--GPU=0', '--NUM_FOLD=10', '--Augment_Data=BIES_bert_base_numberic_agument_data_0.2_200.json'].txt",
                 "performance_['--Data_ID=2', '--GPU=2', '--NUM_FOLD=10', '--Augment_Data=BIES_bert_base_numberic_agument_data_0.3_200.json'].txt",
                 "performance_['--Data_ID=3', '--GPU=2', '--NUM_FOLD=10', '--Augment_Data=BIES_bert_base_numberic_agument_data_0.4_200.json'].txt",
                 "performance_['--Data_ID=4', '--GPU=3', '--NUM_FOLD=10', '--Augment_Data=BIES_bert_base_numberic_agument_data_0.5_200.json'].txt",
                 # "performance_['--Data_ID=5', '--GPU=2', '--NUM_FOLD=10', '--Augment_Data=BIES_bert_base_numberic_agument_data_0.6_200.json'].txt",
    ]

    for file in file_list:
        file = os.path.join(dir, file)
        with open(file, "r") as f:
            data = f.readlines()

        task_list = [12, 3, 4, 5]
        need_lines = 51
        need_line_list = [14,25,36,47]
        task12, task3, task4, task5 = get_list(data, need_lines, task_list, need_line_list)
        assert len(task3) ==10
        assert len(task12) ==10


        merge_task12 = merge_dic(task12)
        print(merge_task12)
        print()
