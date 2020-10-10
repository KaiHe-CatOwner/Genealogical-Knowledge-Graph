import matplotlib.pyplot as plt
from collections import Counter
import nltk
import os
import lxml.etree as etree
import spacy
import tqdm

nlp = spacy.load("en_core_web_sm")

def parse_xml_text(file):
    doc = etree.parse(file)
    text = doc.xpath("//TEXT")[0].text
    return text

def seg_token(sentence):
    # token_list = nltk.word_tokenize(sentence)
    token_list = [str(i) for i in nlp(sentence)]
    return token_list

def sent_len(file):
    with open(file, "r") as f:
        raw_data = f.readlines()

    sentence_len_list = []
    for i in raw_data[1:100]:
        length = len(seg_token(i.split("|")[1]))
        sentence_len_list.append(length)
        # if length == 3:
        #     print(i.split("|")[1])
    print(len(sentence_len_list))

    dic_need = dict(Counter(sentence_len_list))
    print(sum(dic_need.values()))
    print(dic_need.items())

    a = []
    num_0_10 = 0
    num_10_20 = 0
    num_20_30 = 0
    num_30_40 = 0
    num_40_50 = 0
    num_50_60 = 0
    num_60_70 = 0
    num_70_80 = 0
    num_80_90 = 0
    num_90_100 = 0
    num_100_128 = 0
    num_128_more = 0
    for i in dic_need.items():
        if i[0]>0 and i[0]<=10:
            num_0_10+=i[0]
        if i[0]>10 and i[1]<=20:
            num_10_20+=i[1]
        if i[0]>20 and i[0]<=30:
            num_20_30+=i[1]
        if i[0]>30 and i[0]<=40:
            num_30_40+=i[1]
        if i[0]>40 and i[0]<=50:
            num_40_50+=i[1]
        if i[0]>50 and i[0]<=60:
            num_50_60+=i[1]
        if i[0]>60 and i[0]<=70:
            num_60_70+=i[1]
        if i[0]>70 and i[0]<=80:
            num_70_80+=i[1]
        if i[0]>80 and i[0]<=90:
            num_80_90+=i[1]
        if i[0]>90 and i[0]<=100:
            num_90_100+=i[1]
        if i[0]>100 and i[0]<=128:
            num_100_128+=i[1]
        if i[0] > 128:
            num_128_more += i[1]
            a.append(i[0])


    print(num_0_10)
    print(num_10_20)
    print(num_20_30)
    print(num_30_40)
    print(num_40_50)
    print(num_50_60)
    print(num_60_70)
    print(num_70_80)
    print(num_80_90)
    print(num_90_100)
    print('num_100_128', num_100_128)
    print("num_128_more", num_128_more)
    print(a)

def doc_len():
    dir_0 = "../processed_data_Location_ID_corrected"
    target_data_file_path = '../sentence_level_corpus'
    file_list = []
    for file_tuple in os.walk(dir_0):
        for file in file_tuple[2]:
            file = os.path.join(dir_0, file)
            file_list.append(file)

    print("There are %s file !" % str(len(file_list)))

    file_count = 0
    sent_len_list = []
    sent_2 = []
    for file in file_list:
        # file = os.path.join(dir_0, "2943.txt.xml")
        # print(file)
        text = parse_xml_text(file)
        # sentences = spacy_split_sentence(text)
        token_list = seg_token(text)
        sent_len_list.append(len(token_list))
        if len(token_list)>512:
            sent_2.append(len(token_list))

    # print(sent_len_list)
    print(max(sent_len_list))
    print(min(sent_len_list))
    print(len(sent_2))



if __name__ == '__main__':
    file = "../sentence_level_corpus_all_information_normalized_improve_location.csv"
    sent_len(file)
    # doc_len()


