from get_ten_fold import get_10_random_data
import torchtext
import torch
import numpy as np
from threading import Thread
from TP_FN_FP import get_TP_FN_FP
import sys
import os


class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            print("MyThread Exception")


# TASK 1, 2
def only_relationship(example_set):
    for i in range(len(example_set)):
        for j in range(len(example_set[i].tags)):
            if "Person_" in example_set[i].tags[j]:
                example_set[i].tags[j] = example_set[i].tags[j].split("_")[0] + "_" + example_set[i].tags[j].split("_")[
                    1] + "_" + example_set[i].tags[j].split("_")[2]
    return example_set


# TASK 1, 2 (delete name from Location tags)
def location_delete_name(example_set):
    for i in range(len(example_set)):
        for j in range(len(example_set[i].tags)):
            if "Location" in example_set[i].tags[j]:
                example_set[i].tags[j] = example_set[i].tags[j].split("-")[0] + "_" + example_set[i].tags[j].split("_")[
                    -1]
    return example_set


# TASK3
def name_propagation_prepared(example_set):
    for i in range(len(example_set)):
        for j in range(len(example_set[i].tags)):
            if "Person_" in example_set[i].tags[j]:
                example_set[i].tags[j] = example_set[i].tags[j].split("_")[3]
            else:
                example_set[i].tags[j] = "O"
    return example_set


# TASK 4
def parentheses_prepared(example_set):
    for i in range(len(example_set)):
        for j in range(len(example_set[i].tags)):
            if "Person_" in example_set[i].tags[j]:
                if example_set[i].tags[j].split("_")[4] == "None":
                    example_set[i].tags[j] = "O"
                else:
                    example_set[i].tags[j] = example_set[i].tags[j].split("_")[4]
            else:
                example_set[i].tags[j] = "O"
    return example_set


# TASK 5
def person_location_tags_prepared(example_set):
    for i in range(len(example_set)):
        for j in range(len(example_set[i].tags)):
            if example_set[i].tags[j] != "O":
                example_set[i].tags[j] = 1
            else:
                example_set[i].tags[j] = 0
    return example_set


# TASK 5
# Location = location + name,  Name = Person
def person_location_prepared(example_set):
    for i in range(len(example_set)):
        for j in range(len(example_set[i].tags)):
            if "Location" in example_set[i].tags[j]:
                example_set[i].tags[j] = example_set[i].tags[j].split("_")[0]
            elif "Person_" in example_set[i].tags[j]:
                example_set[i].tags[j] = example_set[i].tags[j].split("_")[0]
            elif "DeathName" in example_set[i].tags[j]:
                example_set[i].tags[j] = "Person"
            else:
                example_set[i].tags[j] = "O"
    return example_set


# prepared_data
def filter_content(train_set, test_set, TOEKNS, TAGS, BATCH_SIZE, device,
                   entity_only_flag=False, name_propagation_flag=False, parentheses_flag=False, person_location_flag= False,
                   ):
    if entity_only_flag:
        train_set = location_delete_name(train_set)
        test_set = location_delete_name(test_set)
        train_set = only_relationship(train_set)
        test_set = only_relationship(test_set)
        TAGS.build_vocab(train_set, test_set)
    elif name_propagation_flag:
        train_set = name_propagation_prepared(train_set)
        test_set = name_propagation_prepared(test_set)
        TAGS.build_vocab(train_set, test_set)
    elif parentheses_flag:
        train_set = parentheses_prepared(train_set)
        test_set = parentheses_prepared(test_set)
    elif person_location_flag:
        train_set = person_location_prepared(train_set)
        test_set = person_location_prepared(test_set)

    TAGS.build_vocab(train_set, test_set)

    train_iterator = torchtext.data.Iterator(train_set, batch_size=BATCH_SIZE,shuffle=False,
          repeat=False, device=device)

    test_iterator = torchtext.data.Iterator(test_set, batch_size=BATCH_SIZE, sort=False, shuffle=False, train=False,
         sort_key = lambda x: len(x.tokens), repeat=False, device=device)

    return train_iterator, test_iterator, TOEKNS, TAGS


def establish_file(fold_num, SIMPLFY_NUM, Data_ID, test_flag, bert_model=None):
    if bert_model is None:
        if test_flag:
            # file = "../data/data_BIES/test_test.json"
            file = "../data/data_BIES/test_half.json"
        elif SIMPLFY_NUM == 0:
            file = "../data/data_BIES/BIES.json"
        else:
            file = "../data/data_BIES/simplify_BIES_" + str(SIMPLFY_NUM) + ".json"
    else:
        if bert_model == "tiny":
            if test_flag:
                file = "../data/data_BIES/test_test_bert_tiny_numbric.json"
                # file = "../data/data_BIES/test_half_bert_tiny_numbric.json"
            elif SIMPLFY_NUM == 0:
                file = "../data/data_BIES/BIES_bert_tiny_numberic.json"
            else:
                file = "../data/data_BIES/simplify_BIES_bert_tiny_" + str(SIMPLFY_NUM) + "_numberic.json"
        elif bert_model == "base":
            if test_flag:
                file = "../data/data_BIES/test_test_bert_base_numbric.json"
                # file = "../data/data_BIES/test_half_bert_base_numbric.json"
            elif SIMPLFY_NUM == 0:
                file = "../data/data_BIES/BIES_bert_base_numberic.json"
            else:
                file = "../data/data_BIES/simplify_BIES_bert_base_" + str(SIMPLFY_NUM) + "_numberic.json"


    get_10_random_data(Data_ID, file, fold_num)


def combine_file(raw_file_dir, augment_data_file_dir, file1, augment_data):

    if augment_data is not None:
        new_file = os.path.join(raw_file_dir, "augment_" + str(file1))
        file1 = os.path.join(raw_file_dir, file1)
        augment_data = os.path.join(augment_data_file_dir, augment_data)
        with open(file1, "r") as f:
            data1 = f.readlines()

        with open(augment_data, "r") as f:
            augment_data = f.readlines()

        new_data_list = []
        for i in augment_data:
            dic_temp = {}
            i = i.replace("\'", "\"")
            dic_temp["tokens"] = eval(i)[1]
            dic_temp["tags"] = eval(i)[2]
            new_data_list.append(str(dic_temp).replace("\'", "\"")+"\n")

        with open(new_file, "w") as f:
            f.writelines(data1+new_data_list)

        return os.path.basename(new_file)
    else:
        return file1


def prepared_data(Data_ID, EMBEDDING_DIM, test_50_embeding_flag=False, pad_ID=None, unk_ID=None, bert_flag=False, augment_data=None):

    if not bert_flag:
        if test_50_embeding_flag:
            vectors = 'glove.6B.50d'
        else:
            vectors = 'glove.840B.300d'

        TOEKNS = torchtext.data.Field(lower=True, batch_first=True, pad_token="<PAD>", unk_token="<UNK>")
        TAGS = torchtext.data.Field(dtype=torch.long, batch_first=True, pad_token="<PAD>", unk_token=None)
        fileds = {'tokens': ('tokens', TOEKNS), 'tags': ('tags', TAGS)}  # 字典

        raw_file_dir = '../data/data_BIES/temp_split_data'
        augment_data_file_dir = '../data/data_BIES'

        train_set, test_set = torchtext.data.TabularDataset.splits(path=raw_file_dir,
                                                                   train=combine_file(raw_file_dir,
                                                                                      augment_data_file_dir,
                                                                                      Data_ID + '_train.json',
                                                                                      augment_data),
                                                                   test=Data_ID + '_test.json',
                                                                   format='json',
                                                                   fields=fileds,
                                                                   )
        TOEKNS.build_vocab(train_set, test_set, vectors=vectors,
                           vectors_cache='../../../Data/embedding',
                           unk_init=lambda x: torch.FloatTensor(np.random.uniform(-0.25, 0.25, EMBEDDING_DIM))
                           )
    else:

        TOEKNS = torchtext.data.Field(batch_first=True, use_vocab=False, pad_token=pad_ID, unk_token=unk_ID)
        TAGS = torchtext.data.Field(dtype=torch.long, batch_first=True, pad_token="<PAD>", unk_token=None)
        fileds = {'tokens': ('tokens', TOEKNS), 'tags': ('tags', TAGS)}  # 字典

        raw_file_dir = '../data/data_BIES/temp_split_data'
        augment_data_file_dir = '../data/data_BIES'

        train_set, test_set = torchtext.data.TabularDataset.splits(path=raw_file_dir,
                                                                   train=combine_file(raw_file_dir, augment_data_file_dir,
                                                                                      Data_ID + '_train.json',
                                                                                      augment_data),
                                                                   test=Data_ID + '_test.json',
                                                                   format='json',
                                                                   fields=fileds,
                                                                   )

    return train_set, test_set, TOEKNS, TAGS


# record result
def make_dic_performance(targets, predictions, sentence, sentence_length, TAGS, each_TP_FN_FP,
                         my_improve_Flag, relationship_flag):
    def add_new_kinship(total_return_kin_ship, new_dic):
        for key, value in new_dic.items():
            try:
                total_return_kin_ship[key] = ((total_return_kin_ship[key][0] + value[0]),
                                              (total_return_kin_ship[key][1] + value[1]),
                                              (total_return_kin_ship[key][2] + value[2]))
            except:
                total_return_kin_ship[key] = value

        return total_return_kin_ship

    targets_sentence = targets[sentence][:sentence_length]
    predictions_sentence = predictions[sentence][:sentence_length]
    targets_sentence = [TAGS.vocab.itos[i] for i in targets_sentence.cpu().numpy().tolist()]
    predictions_sentence = [TAGS.vocab.itos[i] for i in predictions_sentence.cpu().numpy().tolist()]

    dic_performance = get_TP_FN_FP(targets_sentence, predictions_sentence, my_improve_Flag, relationship_flag)
    each_TP_FN_FP = add_new_kinship(each_TP_FN_FP, dic_performance)

    return each_TP_FN_FP


def calculate_PRF(each_TP_FN_FP):
    TP_total = 0
    FP_total = 0
    FN_total = 0

    each_PRF = {}
    p_list = []
    r_list = []
    f_list = []

    def return_PRF(TP, FN, FP):
        P = TP / (TP + FP) if (TP + FP) != 0 else 0
        R = TP / (TP + FN) if (TP + FN) != 0 else 0
        F = 2 * P * R / (P + R) if (P + R) != 0 else 0
        if TP == 0 and FN ==0 and FP==0:
            return np.nan, np.nan, np.nan
        return P*100, R*100, F*100

    for key, values in each_TP_FN_FP.items():
        p, r, f = return_PRF(values[0], values[1], values[2])
        TP_total += values[0]
        FN_total += values[1]
        FP_total += values[2]
        each_PRF[key] = (p, r, f)

    for key, values in each_PRF.items():
        p_list.append(values[0])
        r_list.append(values[1])
        f_list.append(values[2])

    P_macro = np.average(p_list)
    R_macro = np.average(r_list)
    F_macro = np.average(f_list)

    P_micro, R_micro, F_micro = return_PRF(TP_total, FN_total, FP_total)
    return P_micro, R_micro, F_micro, TP_total, FN_total, FP_total, p_list, r_list, f_list, P_macro, R_macro, F_macro, each_PRF


# total recored : file = '/performance_total.txt'
def record_each_performance(file, final_result_list_P_macro, final_result_list_R_macro, final_result_list_F_macro,
                            final_result_list_P_micro, final_result_list_R_micro, final_result_list_F_micro,
                            each_fold_average_list, task_List, model_name):

    task12_micro_P_average_list = []
    task12_micro_R_average_list = []
    task12_micro_F_average_list = []
    task12_macro_P_average_list = []
    task12_macro_R_average_list = []
    task12_macro_F_average_list = []

    task3_micro_P_average_list = []
    task3_micro_R_average_list = []
    task3_micro_F_average_list = []
    task3_macro_P_average_list = []
    task3_macro_R_average_list = []
    task3_macro_F_average_list = []

    task4_micro_P_average_list = []
    task4_micro_R_average_list = []
    task4_micro_F_average_list = []
    task4_macro_P_average_list = []
    task4_macro_R_average_list = []
    task4_macro_F_average_list = []

    task5_micro_P_average_list = []
    task5_micro_R_average_list = []
    task5_micro_F_average_list = []
    task5_macro_P_average_list = []
    task5_macro_R_average_list = []
    task5_macro_F_average_list = []

    for dic_result in each_fold_average_list:
        if 12 in task_List:
            task12_micro_P = dic_result[12][0][0]
            task12_micro_R = dic_result[12][0][1]
            task12_micro_F = dic_result[12][0][2]
            task12_macro_P = dic_result[12][1][0]
            task12_macro_R = dic_result[12][1][1]
            task12_macro_F = dic_result[12][1][2]
            task12_micro_P_average_list.append(task12_micro_P)
            task12_micro_R_average_list.append(task12_micro_R)
            task12_micro_F_average_list.append(task12_micro_F)
            task12_macro_P_average_list.append(task12_macro_P)
            task12_macro_R_average_list.append(task12_macro_R)
            task12_macro_F_average_list.append(task12_macro_F)

        if 3 in task_List:
            task3_micro_P = dic_result[3][0][0]
            task3_micro_R = dic_result[3][0][1]
            task3_micro_F = dic_result[3][0][2]
            task3_macro_P = dic_result[3][1][0]
            task3_macro_R = dic_result[3][1][1]
            task3_macro_F = dic_result[3][1][2]
            task3_micro_P_average_list.append(task3_micro_P)
            task3_micro_R_average_list.append(task3_micro_R)
            task3_micro_F_average_list.append(task3_micro_F)
            task3_macro_P_average_list.append(task3_macro_P)
            task3_macro_R_average_list.append(task3_macro_R)
            task3_macro_F_average_list.append(task3_macro_F)

        if 4 in task_List:
            task4_micro_P = dic_result[4][0][0]
            task4_micro_R = dic_result[4][0][1]
            task4_micro_F = dic_result[4][0][2]
            task4_macro_P = dic_result[4][1][0]
            task4_macro_R = dic_result[4][1][1]
            task4_macro_F = dic_result[4][1][2]
            task4_micro_P_average_list.append(task4_micro_P)
            task4_micro_R_average_list.append(task4_micro_R)
            task4_micro_F_average_list.append(task4_micro_F)
            task4_macro_P_average_list.append(task4_macro_P)
            task4_macro_R_average_list.append(task4_macro_R)
            task4_macro_F_average_list.append(task4_macro_F)

        if 5 in task_List:
            task5_micro_P = dic_result[5][0][0]
            task5_micro_R = dic_result[5][0][1]
            task5_micro_F = dic_result[5][0][2]
            task5_macro_P = dic_result[5][1][0]
            task5_macro_R = dic_result[5][1][1]
            task5_macro_F = dic_result[5][1][2]
            task5_micro_P_average_list.append(task5_micro_P)
            task5_micro_R_average_list.append(task5_micro_R)
            task5_micro_F_average_list.append(task5_micro_F)
            task5_macro_P_average_list.append(task5_macro_P)
            task5_macro_R_average_list.append(task5_macro_R)
            task5_macro_F_average_list.append(task5_macro_F)

    with open(file, "a") as f:
        Model_name = str(model_name) + ":  "+str(sys.argv[1:]) + "\n"
        f.writelines(Model_name)

        test_result_averageP_macro = np.average(final_result_list_P_macro)
        test_result_averageR_macro = np.average(final_result_list_R_macro)
        test_result_averageF_macro = np.average(final_result_list_F_macro)
        test_std_P_macro = np.std(final_result_list_P_macro)
        test_std_R_macro = np.std(final_result_list_R_macro)
        test_std_F_macro = np.std(final_result_list_F_macro)

        total_result_macro = "total_result_macro |  averageP %3.2f(%3.2f), averageR %3.2f(%3.2f), averageF %3.2f(%3.2f) \n" % (
            float(test_result_averageP_macro), float(test_std_P_macro),
            float(test_result_averageR_macro), float(test_std_R_macro),
            float(test_result_averageF_macro), float(test_std_F_macro),)
        f.writelines(total_result_macro)

        if 12 in task_List:
            task12_macro_P = np.average(task12_macro_P_average_list)
            task12_macro_R = np.average(task12_macro_R_average_list)
            task12_macro_F = np.average(task12_macro_F_average_list)
            task12_std_P_macro = np.std(final_result_list_P_macro)
            task12_std_R_macro = np.std(final_result_list_R_macro)
            task12_std_F_macro = np.std(final_result_list_F_macro)

            task12_result_macro = "task12_macro |  averageP %3.2f(%3.2f), averageR %3.2f(%3.2f), averageF %3.2f(%3.2f) \n" % (
                float(task12_macro_P), float(task12_std_P_macro),
                float(task12_macro_R), float(task12_std_R_macro),
                float(task12_macro_F), float(task12_std_F_macro),)
            f.writelines(task12_result_macro)

        if 3 in task_List:
            task3_macro_P = np.average(task3_macro_P_average_list)
            task3_macro_R = np.average(task3_macro_R_average_list)
            task3_macro_F = np.average(task3_macro_F_average_list)
            task3_std_P_macro = np.std(final_result_list_P_macro)
            task3_std_R_macro = np.std(final_result_list_R_macro)
            task3_std_F_macro = np.std(final_result_list_F_macro)

            task3_result_macro = "task3_macro |  averageP %3.2f(%3.2f), averageR %3.2f(%3.2f), averageF %3.2f(%3.2f) \n" % (
                float(task3_macro_P), float(task3_std_P_macro),
                float(task3_macro_R), float(task3_std_R_macro),
                float(task3_macro_F), float(task3_std_F_macro),)
            f.writelines(task3_result_macro)

        if 4 in task_List:
            task4_macro_P = np.average(task4_macro_P_average_list)
            task4_macro_R = np.average(task4_macro_R_average_list)
            task4_macro_F = np.average(task4_macro_F_average_list)
            task4_std_P_macro = np.std(final_result_list_P_macro)
            task4_std_R_macro = np.std(final_result_list_R_macro)
            task4_std_F_macro = np.std(final_result_list_F_macro)

            task4_result_macro = "task4_macro |  averageP %3.2f(%3.2f), averageR %3.2f(%3.2f), averageF %3.2f(%3.2f) \n" % (
                float(task4_macro_P), float(task4_std_P_macro),
                float(task4_macro_R), float(task4_std_R_macro),
                float(task4_macro_F), float(task4_std_F_macro),)
            f.writelines(task4_result_macro)

        if 5 in task_List:
            task5_macro_P = np.average(task5_macro_P_average_list)
            task5_macro_R = np.average(task5_macro_R_average_list)
            task5_macro_F = np.average(task5_macro_F_average_list)
            task5_std_P_macro = np.std(final_result_list_P_macro)
            task5_std_R_macro = np.std(final_result_list_R_macro)
            task5_std_F_macro = np.std(final_result_list_F_macro)

            task5_result_macro = "task5_macro |  averageP %3.2f(%3.2f), averageR %3.2f(%3.2f), averageF %3.2f(%3.2f) \n" % (
                float(task5_macro_P), float(task5_std_P_macro),
                float(task5_macro_R), float(task5_std_R_macro),
                float(task5_macro_F), float(task5_std_F_macro),)
            f.writelines(task5_result_macro)

        f.writelines("\n")

        test_result_averageP_micro = np.average(final_result_list_P_micro)
        test_result_averageR_micro = np.average(final_result_list_R_micro)
        test_result_averageF_micro = np.average(final_result_list_F_micro)
        test_std_P_micro = np.std(final_result_list_P_micro)
        test_std_R_micro = np.std(final_result_list_R_micro)
        test_std_F_micro = np.std(final_result_list_F_micro)

        total_result_micro = "total_result_micro |  averageP %3.2f(%3.2f), averageR %3.2f(%3.2f), averageF %3.2f(%3.2f) \n" % (
            float(test_result_averageP_micro), float(test_std_P_micro),
            float(test_result_averageR_micro), float(test_std_R_micro),
            float(test_result_averageF_micro), float(test_std_F_micro),)
        f.writelines(total_result_micro)
        if 12 in task_List:
            task12_micro_P = np.average(task12_micro_P_average_list)
            task12_micro_R = np.average(task12_micro_R_average_list)
            task12_micro_F = np.average(task12_micro_F_average_list)
            task12_std_P_micro = np.std(task12_micro_P_average_list)
            task12_std_R_micro = np.std(task12_micro_R_average_list)
            task12_std_F_micro = np.std(task12_micro_F_average_list)

            task12_result_micro = "task12_micro |  averageP %3.2f(%3.2f), averageR %3.2f(%3.2f), averageF %3.2f(%3.2f) \n" % (
                float(task12_micro_P), float(task12_std_P_micro),
                float(task12_micro_R), float(task12_std_R_micro),
                float(task12_micro_F), float(task12_std_F_micro),)
            f.writelines(task12_result_micro)

        if 3 in task_List:
            task3_micro_P = np.average(task3_micro_P_average_list)
            task3_micro_R = np.average(task3_micro_R_average_list)
            task3_micro_F = np.average(task3_micro_F_average_list)
            task3_std_P_micro = np.std(task3_micro_P_average_list)
            task3_std_R_micro = np.std(task3_micro_R_average_list)
            task3_std_F_micro = np.std(task3_micro_F_average_list)

            task3_result_micro = "task3_micro |  averageP %3.2f(%3.2f), averageR %3.2f(%3.2f), averageF %3.2f(%3.2f) \n" % (
                float(task3_micro_P), float(task3_std_P_micro),
                float(task3_micro_R), float(task3_std_R_micro),
                float(task3_micro_F), float(task3_std_F_micro),)
            f.writelines(task3_result_micro)

        if 4 in task_List:

            task4_micro_P = np.average(task4_micro_P_average_list)
            task4_micro_R = np.average(task4_micro_R_average_list)
            task4_micro_F = np.average(task4_micro_F_average_list)
            task4_std_P_micro = np.std(task4_micro_P_average_list)
            task4_std_R_micro = np.std(task4_micro_R_average_list)
            task4_std_F_micro = np.std(task4_micro_F_average_list)

            task4_result_micro = "task4_micro |  averageP %3.2f(%3.2f), averageR %3.2f(%3.2f), averageF %3.2f(%3.2f) \n" % (
                float(task4_micro_P), float(task4_std_P_micro),
                float(task4_micro_R), float(task4_std_R_micro),
                float(task4_micro_F), float(task4_std_F_micro),)
            f.writelines(task4_result_micro)

        if 5 in task_List:
            task5_micro_P = np.average(task5_micro_P_average_list)
            task5_micro_R = np.average(task5_micro_R_average_list)
            task5_micro_F = np.average(task5_micro_F_average_list)
            task5_std_P_micro = np.std(task5_micro_P_average_list)
            task5_std_R_micro = np.std(task5_micro_R_average_list)
            task5_std_F_micro = np.std(task5_micro_F_average_list)

            task5_result_micro = "task5_micro |  averageP %3.2f(%3.2f), averageR %3.2f(%3.2f), averageF %3.2f(%3.2f) \n" % (
                float(task5_micro_P), float(task5_std_P_micro),
                float(task5_micro_R), float(task5_std_R_micro),
                float(task5_micro_F), float(task5_std_F_micro),)
            f.writelines(task5_result_micro)

    with open(file, "a") as f:
        f.writelines("\n")
        f.writelines("===============================================================")
        f.writelines("\n")


# best result
def record_detail_result(file, test_total, targets_total, predictions_total, TOEKNS_TAGS_dic, Task_list, batch_num, fold_num, test_data_count, bert_flag=False):

    if batch_num ==0 and fold_num==0:
        write_flag = "w"
    else:
        write_flag = "a"

    def TOEKNS_itos(item):
        if bert_flag:
            TOEKNS_itos = TOEKNS_TAGS_dic[0][0](int(item))
        else:
            TOEKNS_itos = TOEKNS_TAGS_dic[0][0].vocab.itos[item]
        return TOEKNS_itos

    with open(file, write_flag) as f:

        if 12 in Task_list:
            test_targets_entity_total = targets_total[12]
            predictions_entity_total = predictions_total[12]
        if 3 in Task_list:
            test_targets_name_propagation_total = targets_total[3]
            predictions_name_propagation_total = predictions_total[3]
        if 4 in Task_list:
            test_targets_parentheses_total = targets_total[4]
            predictions_parentheses_total = predictions_total[4]
        if 5 in Task_list:
            test_targets_person_location_total = targets_total[5]
            predictions_person_location_total = predictions_total[5]

        for sent_index in range(len(test_total)):
                sen_len = 0
                f.write("<sentence>-NO." + str(test_data_count) + "\n")
                for step, item in enumerate(test_total[sent_index]):
                    token = TOEKNS_itos(item)
                    if token == '<PAD>' or token == '[PAD]':
                        break
                    sen_len = step + 1
                    f.write("%4s " % token + "")
                f.write("\n")
                f.write("\n")
                test_data_count += 1

                if 12 in Task_list:
                    f.write("<gold_entity>\n")
                    for item in test_targets_entity_total[sent_index][:sen_len]:
                        tar = TOEKNS_TAGS_dic[12][1].vocab.itos[item]
                        f.write("%6s " % tar + " ")
                    f.write("\n")
                    f.write("\n")

                    f.write("<predictions_entity>\n")
                    for item in predictions_entity_total[sent_index][:sen_len]:
                        tar = TOEKNS_TAGS_dic[12][1].vocab.itos[item]
                        f.write("%6s " % tar + " ")
                    f.write("\n")
                    f.write("\n")

                if 3 in Task_list:
                    f.write("<gold_name_propagation>\n")
                    for item in test_targets_name_propagation_total[sent_index][:sen_len]:
                        tar = TOEKNS_TAGS_dic[3][1].vocab.itos[item]
                        f.write("%6s " % tar + " ")
                    f.write("\n")
                    f.write("\n")

                    f.write("<predictions_name_propagation>\n")
                    for item in predictions_name_propagation_total[sent_index][:sen_len]:
                        tar = TOEKNS_TAGS_dic[3][1].vocab.itos[item]
                        f.write("%6s " % tar + " ")
                    f.write("\n")
                    f.write("\n")

                if 4 in Task_list:
                    f.write("<gold_parentheses>\n")
                    for item in test_targets_parentheses_total[sent_index][:sen_len]:
                        tar = TOEKNS_TAGS_dic[4][1].vocab.itos[item]
                        f.write("%6s " % tar + " ")
                    f.write("\n")
                    f.write("\n")

                    f.write("<predictions_parentheses>\n")
                    for item in predictions_parentheses_total[sent_index][:sen_len]:
                        tar = TOEKNS_TAGS_dic[4][1].vocab.itos[item]
                        f.write("%6s " % tar + " ")
                    f.write("\n")
                    f.write("\n")

                if 5 in Task_list:
                    f.write("<gold_person_location>\n")
                    count_glod = 0
                    str_tar = " "
                    if len(test_targets_person_location_total) > 0:
                        for key, item in test_targets_person_location_total[sent_index].items():
                            if (key != "<PAD>" or key != "[PAD]") and item == 1:
                                str_tar = str_tar + str(count_glod) + ": " + str((key, item)) + "   "
                                count_glod += 1
                        f.write("%6s " % str_tar + " ")
                    else:
                        f.write(" ")
                    f.write("\n")
                    f.write("\n")

                    f.write("<predictions_person_location>\n")
                    count_pred = 0
                    str_pred = ""
                    if len(predictions_person_location_total)>0:
                        for item in predictions_person_location_total[sent_index]:
                            if (item[0] != "<PAD>" or item[0] != "[PAD]") and item[1]==1:
                                str_pred = str_pred + str(count_pred) + ": " + str(item) + "   "
                                count_pred += 1

                        f.write("%6s " % str_pred + " ")
                    else:
                        f.write(" ")

                    f.write("\n")
                    f.write("\n")

                f.write("\n")
                f.write("\n")
                f.write("\n")

    return test_data_count


# each epoch best performance
def write_dic_performance(file, total_micro_result,total_macro_result, dic_reslut, Task_list, fold_num, epoch):

    if fold_num == 0:
        write_flag = "w"
    else:
        write_flag = "a"

    # dic_reslut[i] = [task12_micro_result, task12_macro_result, task12_TP_FN_FP]
    with open(file, write_flag) as fp:

        print("=============== fold_num: "+str(fold_num)+"=============== ", file=fp)
        print("train stop in epoch: "+str(epoch)+" ", file=fp)


        print('P_R_F_macro =', round(total_macro_result[0], 3), "  ", round(total_macro_result[1], 3),  "  ", round(total_macro_result[2], 3), file=fp)
        print('P_R_F_micro =', round(total_micro_result[0], 3), "  ", round(total_micro_result[1], 3),  "  ", round(total_micro_result[2], 3), file=fp)
        print(" ", file=fp)

        if 12 in Task_list:
            print('--------- Task 1,2 ---------', file=fp)
            task12_micro_result, task12_macro_result, task12_TP_FN_FP, task12_each_PRF = dic_reslut[12]
            print('P_R_F_entity_macro =', [round(i, 3) for i in task12_macro_result], file=fp)
            print(" ", file=fp)

            print('P_R_F_entity_micro =', [round(i, 3) for i in task12_micro_result], file=fp)
            print(" ", file=fp)

            print('micro_entity_TP_total =', task12_TP_FN_FP[0], file=fp)
            print('micro_entity_FN_total =', task12_TP_FN_FP[1], file=fp)
            print('micro_entity_FP_total =', task12_TP_FN_FP[2], file=fp)

            print(" ", file=fp)
            print('each_PRF =', task12_each_PRF, file=fp)

            print(" ", file=fp)

        if 3 in Task_list:
            print('--------- Task 3 ---------', file=fp)
            task3_micro_result, task3_macro_result, task3_TP_FN_FP, task3_each_PRF = dic_reslut[3]
            print('P_R_F_name_propagation_macro =', [round(i, 3) for i in task3_macro_result], file=fp)
            print(" ", file=fp)

            print('P_R_F_name_propagation_micro =', [round(i, 3) for i in task3_micro_result], file=fp)
            print(" ", file=fp)

            print('micro_name_propagation_TP_total =', task3_TP_FN_FP[0], file=fp)
            print('micro_name_propagation_FN_total =', task3_TP_FN_FP[1], file=fp)
            print('micro_name_propagation_FP_total =', task3_TP_FN_FP[2], file=fp)

            print(" ", file=fp)
            print('each_PRF =', task3_each_PRF, file=fp)

            print(" ", file=fp)

        if 4 in Task_list:
            print('--------- Task 4 ---------', file=fp)
            task4_micro_result, task4_macro_result, task4_TP_FN_FP, task4_each_PRF = dic_reslut[4]
            print('P_R_F_parentheses_macro =', [round(i, 3) for i in task4_macro_result], file=fp)
            print(" ", file=fp)

            print('P_R_F_parentheses_micro =', [round(i, 3) for i in task4_micro_result], file=fp)
            print(" ", file=fp)

            print('micro_parentheses_TP_total =', task4_TP_FN_FP[0], file=fp)
            print('micro_parentheses_FN_total =', task4_TP_FN_FP[1], file=fp)
            print('micro_parentheses_FP_total =', task4_TP_FN_FP[2], file=fp)

            print(" ", file=fp)
            print('each_PRF =', task4_each_PRF, file=fp)

            print(" ", file=fp)

        if 5 in Task_list:
            print('--------- Task 5 ---------', file=fp)
            task5_micro_result, task5_macro_result, task5_TP_FN_FP, task5_each_PRF = dic_reslut[5]
            print('P_R_F_Name_Location_macro =', [round(i, 3) for i in task5_macro_result], file=fp)
            print(" ", file=fp)

            print('P_R_F_Name_Location_micro =', [round(i, 3) for i in task5_micro_result], file=fp)
            print(" ", file=fp)

            print('micro_Name_Location_TP_total =', task5_TP_FN_FP[0], file=fp)
            print('micro_Name_Location_FN_total =', task5_TP_FN_FP[1], file=fp)
            print('micro_Name_Location_FP_total =', task5_TP_FN_FP[2], file=fp)

            print(" ", file=fp)
            print('each_PRF =', task5_each_PRF, file=fp)

        print(" ", file=fp)
        print(" ", file=fp)
        print(" ", file=fp)


if __name__ == "__main__":
    targets = torch.tensor([[ 1,  1,  1,   0,  0,  0],
                            [ 1,  1, 39,   0,  0,  0],
                            [ 1,  1,  1,   1,  3,  1],
                            [ 1,  1,  1,   0,  0,  0],
                            [ 1,  1, 39,   0,  0,  0],
                            [ 1,  1,  1,   0,  0,  0]])
    predictions = torch.tensor([[40, 40, 40, 10, 10, 93],
                                [43, 40, 40, 11, 11, 11],
                                [40, 40, 40, 40, 40, 40],
                                [43, 40, 40, 11, 11, 93],
                                [40, 40, 40, 16, 11, 93],
                                [43, 43, 43, 11, 11, 93]])

    sentence = 1
    sentence_length =2
    each_TP_FN_FP = {}
    TAGS = {"no": 0, "yes": 1, "<PAD>": 2}
    each_TP_FN_FP = make_dic_performance(targets, predictions, sentence, sentence_length, TAGS, each_TP_FN_FP,
                         False, False)


    # # P_micro, R_micro, F_micro, TP_total, FN_total, FP_total, p_list, r_list, f_list, P_macro, R_macro, F_macro
    # each_TP_FN_FP = {'yes': (4, 4, 0), 'no': (114, 4, 0)}
    # task5_flag = False
    # # each_TP_FN_FP["L-N"] = [0,0,0]
    # # task5_flag = True
    # res = calculate_PRF(each_TP_FN_FP, task5_flag)
    # print(res)