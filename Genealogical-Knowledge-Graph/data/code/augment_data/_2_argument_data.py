
from _1_arguement_method import generated_data
import collections
from sklearn.preprocessing import MinMaxScaler, normalize, Normalizer
import numpy as np
import random
from tqdm import tqdm
import re


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


def get_generated_count(old_kinship_dic, total_sentence, generated_ratio, generate_thrshold):
    kinship_list = []
    kinship_number_list = []
    generate_num = int(total_sentence*generated_ratio)
    dic_generated_number = {}
    for k,v in sorted(old_kinship_dic.items(), key=lambda x:x[1], reverse=True):
        if v < generate_thrshold:
            kinship_list.append(k)
            v = np.log(v+1)
            kinship_number_list.append(v)

    print("generated "+ str(generate_num) + " data!")

    # weights = (np.array(kinship_number_list) - min(np.array(kinship_number_list))) / (max(np.array(kinship_number_list)) - min(np.array(kinship_number_list)))
    # weights = np.log((np.array(kinship_number_list)  / sum(np.array(kinship_number_list)))+1)
    weights = (np.array(kinship_number_list) / sum(np.array(kinship_number_list)))
    weights = np.flipud(weights / sum(weights))
    # weights = 1 / weights

    for index in range(len(kinship_list)):
        dic_generated_number[kinship_list[index]] = int(generate_num * (weights[index]))

    temp = dict()
    for key in dic_generated_number.keys() | old_kinship_dic.keys():
        temp[key] = sum([d.get(key, 0) for d in (dic_generated_number, old_kinship_dic)])

    # print(sorted(old_kinship_dic.items(), key=lambda x:x[1], reverse=True))
    # print(sorted(dic_generated_number.items(), key=lambda x:x[1], reverse=True))
    # print(sorted(temp.items(), key=lambda x:x[1], reverse=True))

    return dic_generated_number


#selected_triple = (kinship, kinship's pos), raw_data = raw annotation
def get_kinship_annotation(new_kinship, raw_anno, raw_sentence, gen_new_sentene):
    global Error_count
    new_annotation_list = []

    if raw_anno == "":
        return ""

    raw_anno = eval(raw_anno)
    if type(raw_anno) == list:
        if len(raw_anno) > 0:
            for item in raw_anno:
                anno_span_B, anno_span_E = item[1].split("~")
                anno = raw_sentence[int(anno_span_B): int(anno_span_E)]

                anno = re.sub("\(", r"\(", anno)
                anno = re.sub("\)", r"\)", anno)
                anno = re.sub("\?", r"\?", anno)
                anno = re.sub("\+", r"\+", anno)
                anno = re.sub("\$", r"\$", anno)
                anno = re.sub("\[", r"\[", anno)
                anno = re.sub("\]", r"\]", anno)
                anno = re.sub("\*", r"\*", anno)
                anno = re.sub("\.", r"\.", anno)

                res = re.search(anno, gen_new_sentene)
                if res is not None:
                    new_span = res.span()
                    if len(item) == 4:
                        new_item = (item[0], str(new_span[0])+"~"+str(new_span[1]), item[2], item[3])
                    else:
                        raise Exception("type error!~~")
                    new_annotation_list.append(new_item)
                else:
                    Error_count += 1

        return str(new_annotation_list)
    else:
        return ""

def get_other_annotation(raw_anno, raw_sentence, gen_new_sentene):
    global Error_count

    if raw_anno == "":
        return ""

    raw_anno = eval(raw_anno)
    if type(raw_anno) == tuple:
        anno_span_B, anno_span_E = raw_anno[1].split("~")
        anno = raw_sentence[int(anno_span_B): int(anno_span_E)]

        anno = re.sub("\(", r"\(", anno)
        anno = re.sub("\)", r"\)", anno)
        anno = re.sub("\?", r"\?", anno)
        anno = re.sub("\+", r"\+", anno)
        anno = re.sub("\$", r"\$", anno)
        anno = re.sub("\[", r"\[", anno)
        anno = re.sub("\]", r"\]", anno)
        anno = re.sub("\*", r"\*", anno)
        anno = re.sub("\.", r"\.", anno)

        res = re.search(anno, gen_new_sentene)
        new_item = ""
        if res is not None:
            new_span = res.span()
            if len(raw_anno) == 2:
                new_item = (raw_anno[0], str(new_span[0]) + "~" + str(new_span[1]))
            else:
                raise Exception("type error!~~")
        else:
            Error_count += 1

        return str(new_item)

    if type(raw_anno) == list:
        new_annotation_list = []
        for item in raw_anno:
            anno_span_B, anno_span_E = item[1].split("~")
            anno = raw_sentence[int(anno_span_B): int(anno_span_E)]

            anno = re.sub("\(", r"\(", anno)
            anno = re.sub("\)", r"\)", anno)
            anno = re.sub("\?", r"\?", anno)
            anno = re.sub("\+", r"\+", anno)
            anno = re.sub("\$", r"\$", anno)
            anno = re.sub("\[", r"\[", anno)
            anno = re.sub("\]", r"\]", anno)
            anno = re.sub("\*", r"\*", anno)
            anno = re.sub("\.", r"\.", anno)

            res = re.search(anno, gen_new_sentene)
            if res is not None:
                new_span = res.span()
                if len(item) == 2:
                    new_item = (item[0], str(new_span[0])+"~"+str(new_span[1]))
                else:
                    raise Exception("type error!~~")
                new_annotation_list.append(new_item)
            else:
                Error_count += 1

        return str(new_annotation_list)


def generate_data(dic_generated_number, generated_file, raw_data_list, n):
    generate_list = []

    for new_kinship, num in tqdm(dic_generated_number.items()):
        # print("new_kinship is", new_kinship)
        # print("new_kinship num is", num)

        # temp_list = for each kinship, generated data's list
        temp_list = []
        while len(temp_list) < num:
            # allow make one sentence more than once
            pick_data_index = random.randint(0, len(raw_data_list))
            try:
                kinship_list = eval(raw_data_list[pick_data_index].split("|")[-1])
            except:
                continue

            if len(kinship_list) > 0 and (new_kinship not in kinship_list) and (kinship_list[0][0] in raw_data_list[pick_data_index].split("|")[1]):
                temp_list.append(raw_data_list[pick_data_index])

        for i in temp_list:

            data_list = i.split("|")
            gen_new_sentene, selected_triple = generated_data(new_kinship, data_list, n)
            old_kinship = selected_triple[0]

            gen_new_data = data_list[0] + "|" + gen_new_sentene
            raw_sent = data_list[1]
            # need modify annotation span (Death_Name | Death_Date | Birth_Date | Age_ID | Location_ID | relationship)
            for i in range(2, 7):
                raw_anno = data_list[i]
                new_anno = get_other_annotation(raw_anno, raw_sent, gen_new_sentene)
                gen_new_data = gen_new_data + "|" + new_anno

            raw_anno = data_list[7]
            new_anno = get_kinship_annotation(new_kinship, raw_anno, raw_sent, gen_new_sentene)
            new_anno = new_anno.replace(old_kinship, new_kinship)
            gen_new_data = gen_new_data + "|" + new_anno + "\n"
            generate_list.append(gen_new_data)

    with open(generated_file, "w") as f:
        f.write("Obituary_ID|Sentence|Death_Name|Death_Date|Birth_Date|Age_ID|Location_ID|relationship" + '\n')
        f.writelines(generate_list)


if __name__ == "__main__":
    global Error_count
    Error_count = 0

    generated_ratio = 0.7
    generate_thrshold = 200
    # alpha_sr, alpha_ri, alpha_rs, alpha_rd
    n = [0.1, 0.1, 0.1, 0.1]

    raw_file = '../../sentence_level_corpus_all_information_normalized_improve_location.csv'
    generated_file = '../../agument_data_'+str(generated_ratio)+'_'+str(generate_thrshold)+'.csv'

    with open(raw_file, "r") as f:
        raw_data_list = f.readlines()[1:]
    total_sentence = len(raw_data_list)

    old_kinship_dic = get_kinship_dict(raw_file)
    print(len(old_kinship_dic))
    print(old_kinship_dic)

    dic_generated_number = get_generated_count(old_kinship_dic, total_sentence, generated_ratio, generate_thrshold)
    print(dic_generated_number)

    generate_data(dic_generated_number, generated_file, raw_data_list, n)
    print(Error_count)