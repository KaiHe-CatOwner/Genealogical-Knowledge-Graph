import random
import json

def get_10_random_data(Data_ID, file, fold_num=0):
    print("spliting data ...")
    with open(file, "r") as f:
        data = f.readlines()

    # random.shuffle(data)

    data_index_list = list(range(len(data)))
    need_len = int(data_index_list[-1]*0.1)

    test_set_index_list = data_index_list[need_len*fold_num:need_len*(fold_num+1)]
    for i in test_set_index_list:
        data_index_list.remove(i)

    count = 0
    while len(test_set_index_list) < need_len:
        test_set_index_list.append(count)
        data_index_list.remove(count)
        count +=1


    train_set_index_list = data_index_list
    test_writen_list = []
    for i in test_set_index_list:
        tokens_list = []
        for j in eval(data[i])[1]:
            tokens_list.append(j)
        tags_list = []
        for j in eval(data[i])[2]:
            tags_list.append(j)
        tem_data = {"tokens":tokens_list, "tags":tags_list}
        test_writen_list.append(tem_data)
        # test_writen_list.append("\n")

    with open('../data/data_BIES/temp_split_data/'+Data_ID+'_test.json', "w") as f:
        # f.writelines(test_writen_list)
        for i in test_writen_list:
            f.write(json.dumps(i))
            f.write("\n")



    train_writen_list = []
    for i in train_set_index_list:
        tokens_list = []
        for j in eval(data[i])[1]:
            tokens_list.append(j)
        tags_list = []
        for j in eval(data[i])[2]:
            tags_list.append(j)
        tem_data = {"tokens":tokens_list, "tags":tags_list}
        train_writen_list.append(tem_data)
        # train_writen_list.append("\n")

    with open('../data/data_BIES/temp_split_data/'+Data_ID+'_train.json', "w") as f:
        # f.writelines(train_writen_list)
        for i in train_writen_list:
            f.write(json.dumps(i))
            f.write("\n")


if __name__ == "__main__":
    Data_ID = "test_set"
    fold_num = 10
    get_10_random_data(Data_ID, "../data/data_BIES/test_test.json", fold_num)
