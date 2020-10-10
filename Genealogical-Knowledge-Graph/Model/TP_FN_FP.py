from collections import Counter




# this function return micro average measures, but we need macro average!!!!
def get_TP_FN_FP(targets_sentence, predictions_sentence, improve_flag, kinship_flag=True):
    sen_len = 0
    for i in targets_sentence:
        if i != "<PAD>":
            sen_len+=1

    gold_triple_list = get_triple_O_seg(targets_sentence[:sen_len])
    # print("gold_triple_list", gold_triple_list)
    predicted_triple_list = get_triple_O_seg(predictions_sentence[:sen_len])
    # print("predicted_triple_list", predicted_triple_list)
    if kinship_flag:
        if improve_flag:
            predicted_triple_list = improved_result(predicted_triple_list)
        else:
            predicted_triple_list = formatted_outpus(predicted_triple_list)


    predicted_dic = {}
    for temp_list in predicted_triple_list:
        if kinship_flag:
            relation = temp_list[0][1][:-2]
        else:
            relation = temp_list[0][1]
        try:
            if predicted_dic[relation]:
                new_predicted_value = []
                for i in temp_list:
                    new_predicted_value.append(str(i[0]))
                temp = predicted_dic[relation]
                temp.append(new_predicted_value)
                predicted_dic[relation] = temp
        except:
            new_predicted_value = []
            for i in temp_list:
                new_predicted_value.append(str(i[0]))
            predicted_dic[relation] = [new_predicted_value]

    gold_dic = {}
    for temp_list in gold_triple_list:     # gold_triple_list = [[(2, 'no')]]
        if kinship_flag:
            relation = temp_list[0][1][:-2]
        else:
            relation = temp_list[0][1]
        try:
            if gold_dic[relation]:
                new_value = []
                for i in temp_list:
                    new_value.append(str(i[0]))
                temp = gold_dic[relation]
                temp.append(new_value)
                gold_dic[relation] = temp
        except:
            new_value = []
            for i in temp_list:
                new_value.append(str(i[0]))
            gold_dic[relation] = [new_value]

    dic_kinship_performance = {}

    # get TP, FP
    for relation_type, predicted_result_list in predicted_dic.items():
        TP = 0
        FN = 0
        FP = 0

        try:
            gold_result_list = gold_dic[relation_type]
        except:
            gold_result_list = []

        if len(gold_result_list) == 0:
            FP += len(predicted_result_list)

        for prediction in predicted_result_list:
            if prediction in gold_result_list:
                TP += 1
            elif len(gold_result_list) > 0:
                FP += 1

        dic_kinship_performance[relation_type] = (TP, FN, FP)

    # get FN
    for relation_type, gold_result_list in gold_dic.items():
        try:
            TP, FN, FP = dic_kinship_performance[relation_type]
        except:
            TP, FN, FP = (0, 0, 0)
        try:
            predictions_list = predicted_dic[relation_type]
        except:
            predictions_list = []

        for gold in gold_result_list:
            if gold not in predictions_list:
                FN += 1
                dic_kinship_performance[relation_type] = (TP, FN, FP)

    return dic_kinship_performance


def add_new_kinship(total_return_kin_ship, new_dic):
    for key, value in new_dic.items():
        try:
            temp_List = total_return_kin_ship[key]
            temp_List = temp_List + value
            total_return_kin_ship[key] = set(temp_List)
        except:
            total_return_kin_ship[key] = value

    return total_return_kin_ship


# batch_gold_list : list= [{one stence(key:value)},{one stence} ]  4 sentence
# perdictio list([1, 10, 1])   [ [one stence: (key:value)], []]
# <class 'list'>: [[('.-is', 1), ('.-survivedby', 1), ('.-,stanley', 1), ('.-dee', 1), ('.-shirleyann)', 1), ('.-of', 1), ('.-herbrother-', 1), ('.-,', 1), ('.-maxon', 1), ('.-tx', 1)]]
def make_task5_TP_FN_FP_sentence(batch_gold_persion_location_list, perdiction_persion_location_pair, sentence,
                                 each_person_location_TP_FN_FP):
    TP = each_person_location_TP_FN_FP["L-N"][0]
    FN = each_person_location_TP_FN_FP["L-N"][1]
    FP = each_person_location_TP_FN_FP["L-N"][2]

    if batch_gold_persion_location_list == [] and perdiction_persion_location_pair == []:
        return each_person_location_TP_FN_FP
    else:
        pair_pred_list = []
        targets_sentence_dict = batch_gold_persion_location_list[sentence]
        try:
            predictions_sentence_list = perdiction_persion_location_pair[sentence]
        except:
            predictions_sentence_list = []

        for pair_pred in predictions_sentence_list:
            if pair_pred[0] != "<PAD>":
                pair_pred_list.append(pair_pred[0])
                if pair_pred[1] == 1:
                    try:
                        if targets_sentence_dict[pair_pred[0]] == 1:
                            TP += 1
                        elif targets_sentence_dict[pair_pred[0]] == 0:
                            FP += 1
                    except:
                        # in pred, not in gold
                        FP += 1
                elif pair_pred[1] == 0:
                    try:
                        if targets_sentence_dict[pair_pred[0]] == 1:
                            FN += 1
                    except:
                        # pred=no, not in gold
                        pass

        for key, value in targets_sentence_dict.items():
            if key != "<PAD>":
                if value == 1 and key not in pair_pred_list:
                    FN += 1

    each_person_location_TP_FN_FP["L-N"] = (TP, FN, FP)
    return each_person_location_TP_FN_FP


def formatted_outpus(triple_list_list):
    new_triple_list_list = []
    for triple_list in triple_list_list:
        add_flag = True
        # add only single tag when it is S
        if len(triple_list) == 1:
            if triple_list[0][1][-1] is "S":
                new_triple_list_list.append(triple_list)

        # if one tag of entity is wrong, delete the whole entity
        if len(triple_list) > 1:
            temp = triple_list[0][1].split("_")
            tag = temp[0] + "_" + temp[1]
            for triple_index in range(0, len(triple_list)):
                temp1 = triple_list[triple_index][1].split("_")
                temp_tag = temp1[0] + "_" + temp1[1]
                if temp_tag != tag:
                    add_flag = False
                    break

            # tag must satisfy BIES
            if add_flag:
                for triple_index in range(1, len(triple_list) - 1):
                    if triple_list[triple_index][-1][-1] is not "I":
                        add_flag = False

            # tag must satisfy BIES
            if add_flag:
                if (triple_list[0][-1][-1] != "B") or (triple_list[-1][-1][-1] != "E"):
                    add_flag = False

            if add_flag:
                new_triple_list_list.append(triple_list)

    # print(new_triple_list_list)
    return new_triple_list_list

# delete triple with different tags, inherently can format triple into BIE and different tag.
def improved_result(triple_list_list):
    new_triple_list_list = []
    for triple_list in triple_list_list:
        new_triple_list = []
        index_list = []
        tag_list = []
        pos_list = []

        if len(triple_list) == 1:
            new_triple_list_list.append(triple_list)

        if len(triple_list) > 1:
            # if tag is wrong, fix it
            for triple_index in range(0, len(triple_list)):
                index_list.append(triple_list[triple_index][0])
                tag_list.append(triple_list[triple_index][1][:-2])
                pos_list.append(triple_list[triple_index][1][-1])

            cnt = Counter(tag_list)
            if len(cnt) >1:
                tag_list = [cnt.most_common(1)[0][0]] * len(index_list)

            assert len(index_list) == len(tag_list)
            assert len(pos_list) == len(tag_list)

            for i in range(len(index_list)):
                new_triple_list.append((index_list[i], tag_list[i]+"_"+pos_list[i]))

            ##===========================================
            # tag must satisfy BIES
            # if add_flag:
            #     for triple_index in range(1, len(triple_list) - 1):
            #         if triple_list[triple_index][-1][-1] is not "I":
            #             add_flag = False
            #
            # # tag must satisfy BIES
            # if add_flag:
            #     if (triple_list[0][-1][-1] != "B") or (triple_list[-1][-1][-1] != "E"):
            #         add_flag = False
            ##===========================================

            new_triple_list_list.append(new_triple_list)

    # print(new_triple_list_list)
    return new_triple_list_list

# # delete triple with different tags, inherently can format triple into BIE.
# def improved_result(triple_list_list):
#     new_triple_list_list = []
#     new_triple_list = []
#     add_flag = True
#     for triple_list in triple_list_list:
#         if len(triple_list) == 1:
#             new_triple_list_list.append(triple_list)
#
#         if len(triple_list) > 1:
#             # if tag is wrong, delete the whole entity
#             tag = triple_list[0][1].split("_")[0]
#             for triple_index in range(0, len(triple_list)):
#                 if triple_list[triple_index][1].split("_")[0] != tag:
#                     add_flag = False
#                     break
#
#             if add_flag:
#                 new_triple_list_list.append(triple_list)
#
#     # print(new_triple_list_list)

def improved_result_old(triple_list_list):
    new_triple_list_list = []
    for triple_list in triple_list_list:
        new_triple_list = []
        if len(triple_list) == 1:
            if triple_list[0][1][-1] is not "S":
                new_triple_item = triple_list[0][1][:-1] + "S"
                new_triple_list.append((triple_list[0][0], new_triple_item))
                new_triple_list_list.append(new_triple_list)
            else:
                new_triple_list_list.append(triple_list)

        if len(triple_list) > 1:
            if triple_list[0][-1][-1] is not "B":
                new_triple_item = triple_list[0][-1][:-1] + "B"
                new_triple_list.append((triple_list[0][0], new_triple_item))
            else:
                new_triple_list.append(triple_list[0])

            for triple_index in range(1, len(triple_list) - 1):
                if triple_list[triple_index][-1][-1] is not "I":
                    new_triple_item = triple_list[triple_index][-1][:-1] + "I"
                    new_triple_list.append((triple_list[triple_index][0], new_triple_item))
                else:
                    new_triple_list.append(triple_list[triple_index])

            if triple_list[-1][-1][-1] is not "E":
                new_triple_item = triple_list[-1][-1][:-1] + "E"
                new_triple_list.append((triple_list[-1][0], new_triple_item))
            else:
                new_triple_list.append(triple_list[-1])

            new_triple_list_list.append(new_triple_list)

    # print(new_triple_list_list)
    return new_triple_list_list


# this Function ask the last tag must be "O"
def get_triple_O_seg(targets_sentence):
    total_list = []
    gold_list = []
    for index in range(len(targets_sentence)):
        if targets_sentence[index] == "O":
            if len(gold_list) > 0:
                total_list.append(gold_list)
                gold_list = []
        else:
            if targets_sentence[index] != "<PAD>" and targets_sentence[index] != "no":
                temp_list = targets_sentence[index].split("_")
                if len(temp_list) == 2:
                    add_str = temp_list[0]+"_"+temp_list[0]+"_"+temp_list[1]
                else:
                    add_str = targets_sentence[index]
                gold_list.append((index, add_str))

                if index == len(targets_sentence)-1:
                    total_list.append(gold_list)

    return total_list


def return_PRF(TP, FN, FP):
    P = TP / (TP + FP) if (TP + FP) != 0 else 0
    R = TP / (TP + FN) if (TP + FN) != 0 else 0
    F = 2 * P * R / (P + R) if (P + R) != 0 else 0
    return P, R, F


if __name__ == "__main__":

    # TASK 1 2
    # target_sentence = ['O', 'O', 'Person_spouse_B', 'Person_spouse_I', 'Person_spouse_I', 'Person_spouse_E']
    # predictions_sentence = ['O', 'O', 'Person_spouse_B', 'Person_spouse_I', 'Person_spouse_S', 'O']

    # target_sentence =      ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'Person_wife_B', 'Person_wife_E', 'O', 'O', 'O', 'O', 'O', 'Person_daughter_S', 'O', 'O', 'O', 'Location_S', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    # predictions_sentence = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'Person_wife_B', 'Person_wife_E', 'O', 'O', 'O', 'O', 'O', 'Person_daughter_S', 'O', 'O', 'O', 'O',           'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

    # target_sentence =      ['DeathName_B', 'DeathName_I', 'DeathName_I', 'DeathName_I', 'DeathName_I', 'DeathName_E', 'O', 'Age_S', 'O', 'O', 'Location_S', 'O', 'O', 'O', 'O', 'DeathDate_B', 'DeathDate_I', 'DeathDate_I', 'DeathDate_E', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    # predictions_sentence = ['DeathName_B', 'DeathName_I', 'DeathName_I', 'O', 'O', 'DeathName_E', 'O', 'Age_S', 'O', 'O', 'Location_S', 'O', 'O', 'O', 'O', 'DeathDate_B', 'DeathDate_I', 'DeathDate_I', 'DeathDate_E', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    #
    # target_sentence =      ['DeathName_B', 'DeathName_I', 'DeathName_I', 'DeathName_E', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'Age_B', 'Age_E', 'O', 'O', 'O', 'O', 'O', 'O']
    # predictions_sentence = ['DeathName_B', 'DeathName_I', 'DeathName_I', 'DeathName_E', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'Age_B', 'Age_E', 'O', 'O', 'O', 'O', 'O', 'O']
    #
    # target_sentence =      ['O', 'Person_son_S', 'O', 'O', 'Person_son_B', 'Person_son_I', 'Person_son_I', 'Person_son_I', 'Person_son_E', 'O', 'Location_B', 'Location_I', 'Location_E', 'O', 'O', 'Person_son_B', 'Person_son_I', 'Person_son_I', 'Person_son_I', 'Person_son_E', 'O', 'Location_B', 'Location_I', 'Location_E', 'O', 'O', 'Person_son_B', 'Person_son_E', 'O', 'Location_B', 'Location_I', 'Location_E', 'O', 'O', 'Person_daughter_B', 'Person_daughter_I', 'Person_daughter_I', 'Person_daughter_I', 'Person_daughter_E', 'O', 'Location_B', 'Location_I', 'Location_E', 'O', 'O', 'O', 'O', 'O', 'O', 'Person_son-in-law_B', 'Person_son-in-law_I', 'Person_son-in-law_E', 'O', 'Location_B', 'Location_I', 'Location_E', 'O']
    # predictions_sentence = ['O', 'Person_son_B', 'O', 'O', 'Person_son_B', 'Person_son_I', 'Person_son_I', 'Person_son_I', 'Person_son_E', 'O', 'Location_B', 'Location_I', 'Location_I', 'O', 'O', 'Person_son_B', 'Person_son_I', 'Person_son_I', 'Person_son_I', 'Person_son_E', 'O', 'Location_B', 'Location_I', 'Location_E', 'O', 'O', 'Person_son_B', 'Person_son_E', 'O', 'Location_B', 'Location_I', 'Location_E', 'O', 'O', 'Person_daughter_B', 'Person_daughter_I', 'Person_daughter_I', 'Person_daughter_I', 'Person_daughter_E', 'O', 'Location_B', 'Location_I', 'Location_E', 'O', 'O', 'O', 'O', 'O', 'O', 'Person_son-in-law_B', 'Person_son-in-law_I', 'Person_son-in-law_E', 'O', 'Location_B', 'Location_I', 'Location_E', 'O']
    #
    # TASK 3
    # target_sentence =  ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'no', 'no', 'no', 'no', 'no', 'O', 'O', 'O', 'O', 'O', 'O', 'no', 'no', 'no', 'no', 'no', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'yes', 'O', 'yes', 'yes', 'O']
    # predictions_sentence = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'no', 'no', 'no', 'no', 'no', 'O', 'O', 'O', 'O', 'O', 'O', 'no', 'no', 'no', 'no', 'no', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'yes', 'O', 'yes', 'yes', 'O']

    # target_sentence =       ['no', 'O', 'O',     'no',  'O', 'O', 'yes', 'O', 'yes', '<PAD>', 'O', '<PAD>', 'O']
    # predictions_sentence =  ['yes', 'O', 'yes',  'no',  'no', 'O', 'yes', 'O', 'no', '<PAD>', 'O', '<PAD>', 'O']

    # target_sentence =       ['no', 'O', 'O',   'no', 'O', 'O', 'yes', 'O', 'yes', '<PAD>', '<PAD>']
    # predictions_sentence = ['yes', 'O', 'yes', 'no', 'no', 'O', 'yes', 'O', 'yes',  '<PAD>', '<PAD>']

    # target_sentence = ['O', 'yes', 'O', 'O', 'O', 'O', 'O', '<PAD>']
    # predictions_sentence =['O', 'no', 'O', 'O', 'O', 'O', 'O', '<PAD>']
    # #
    # target_sentence = ['O', 'O', 'no', 'no', 'O', 'O', 'O']
    # predictions_sentence = ['O', 'O', 'yes', 'yes', 'O', 'O', 'O']
    #
    # target_sentence = ['O', 'O', 'no', 'no', 'O', 'O', 'O']
    # predictions_sentence = ['O', 'O', 'O', 'no', 'O', 'O', 'O']
    #
    # TASK4
    # target_sentence =  ['O', 'O', 'O', 'coupled-name', 'coupled-name', 'coupled-name', 'coupled-name', 'coupled-name', 'O', 'nickname', 'nickname', 'nickname', 'nickname', 'nickname', 'O', 'previous-last-name', 'previous-last-name', 'previous-last-name', 'previous-last-name', 'previous-last-name']
    # predictions_sentence = ['O', 'O', 'O', 'coupled-name', 'coupled-name', 'coupled-name', 'coupled-name', 'coupled-name', 'O', 'nickname', 'nickname', 'nickname', 'nickname', 'nickname', 'O', 'previous-last-name', 'previous-last-name', 'previous-last-name', 'previous-last-name', 'previous-last-name']
    #
    # target_sentence =      ['O', 'O', 'previous-last-name', 'previous-last-name', 'previous-last-name', 'previous-last-name', 'previous-last-name', 'O', 'O', 'O']
    # predictions_sentence = ['O', 'O', 'nickname', 'nickname', 'nickname', 'nickname', 'nickname', 'O', 'O', 'O']
    #
    # target_sentence = ['O', 'O', 'previous-last-name', 'previous-last-name', 'previous-last-name', 'previous-last-name', 'previous-last-name', 'O', 'O', 'O', '<PAD>']
    # predictions_sentence = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', '<PAD>']
    #
    # target_sentence = ['O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O']
    # predictions_sentence = ['O', 'O', 'previous-last-name', 'previous-last-name', 'previous-last-name', 'previous-last-name', 'previous-last-name', 'O', 'O', 'O']
    #
    # target_sentence = ['O', 'O', 'previous-last-name', 'previous-last-name', 'previous-last-name', 'previous-last-name', 'previous-last-name', 'O', 'O', 'O']
    # predictions_sentence = ['O', 'O', 'previous-last-name', 'previous-last-name', 'previous-last-name', 'previous-last-name', 'O', 'O', 'O', 'O', 'O']


    # dic_kinship_performance = get_TP_FN_FP(target_sentence, predictions_sentence, improve_flag=True, kinship_flag=True)
    # print(dic_kinship_performance)



    # TASK5

    target_sentence = [{'<PAD>': 2, 'rochester-donovandavidmoonjr.': 1},  {'rochester-donovandavidmoonjr.': 1}]
    predictions_sentence = [[('<PAD>', 1)], [('rochester-donovandavidmoonjr.', 1)]]

    # target_sentence = []
    # predictions_sentence = []

    dic_task5 = make_task5_TP_FN_FP_sentence(target_sentence, predictions_sentence, 0, each_person_location_TP_FN_FP = {"L-N": (0, 0, 0)})
    print(dic_task5)

