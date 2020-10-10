import collections


def get_kinship_dict(file):
    with open(file, 'r') as f:
        raw_data = f.readlines()

    tags_total = []
    for i in raw_data[1:]:
        items = i.split('|')
        try:
            for j in eval(items[-1]):
                tags_total.append(j[0])
        except:
            pass
    kinship_dict = collections.Counter(tags_total)
    return kinship_dict

kinship_list = get_kinship_dict('../sentence_level_corpus_all_information_normalized.csv')

need_num = 10
total_count = 0
needed_list = []
unneeded_list = []
for i in kinship_list.keys():
    total_count += kinship_list[i]
    if kinship_list[i] > need_num:
        needed_list.append(i)
    else:
        unneeded_list.append(i)

print("total", total_count)
print("normalized_kinship_num", len(kinship_list))
print("needed_list", len(needed_list))
print("unneeded_list", len(unneeded_list))
print(unneeded_list)

in_law_count = 0
for i in unneeded_list:
    if "in-law" in i:
        in_law_count+=1
print("in_law relation in delete: ", in_law_count)


def save_result():
    # file = "../data_BIES/BIES.json"
    # taget_file = "../data_BIES/simplify_BIES_"+str(need_num)+".json"

    # file = "../data_BIES/BIES_bert_tiny_numberic.json"
    # taget_file = "../data_BIES/simplify_BIES_bert_tiny_" + str(need_num) + "_numberic.json"

    file = "../data_BIES/BIES_bert_base_numberic.json"
    taget_file = "../data_BIES/simplify_BIES_bert_base_" + str(need_num) + "_numberic.json"

    # taget_file = "../../../data_BIES/Model_data/simplify_BIES_"+str(need_num)+".json"
    # file = "./Model_data/BIES_test.json"
    # taget_file = "./Model_data/simplify_BIES_test.json"

    with open(file, "r") as f:
        data = f.readlines()


    total_list = []
    for item in data:
        item = eval(item)
        data_ID = item[0]
        data_sentence = item[1]
        data_tags = item[2]
        data_new_tags = []
        total = []
        for tag in data_tags:
            if "Person_" in tag:
                tag1 = tag.split("_")[1]
                if tag1 in unneeded_list:
                    data_new_tags.append("O")
                else:
                    data_new_tags.append(tag)
            else:
                data_new_tags.append(tag)
        total.append(data_ID)
        total.append(data_sentence)
        total.append(data_new_tags)
        total_list.append(total)

    with open(taget_file, "w") as f:
        for i in total_list:
            f.writelines(str(i)+"\n")


save_result()