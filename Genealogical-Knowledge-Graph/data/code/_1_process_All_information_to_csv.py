import lxml.etree as etree
import os
import re

def sentence_level_span(annotation_span_list, sen_span):
    sentence_start = sen_span.split("~")[0]
    sentence_level_annotation_span_list = []
    for i in annotation_span_list:
        span_list = i[1]
        start_pos, end_pos = span_list.split("~")
        if len(i) == 2:
            sentence_level_annotation_span_list.append((i[0], str(int(start_pos)-int(sentence_start))+"~"+str(int(end_pos)-int(sentence_start)), "", ""))
        elif len(i) ==4:
            sentence_level_annotation_span_list.append((i[0], str(int(start_pos)-int(sentence_start))+"~"+str(int(end_pos)-int(sentence_start)), i[2], i[3]))
        else:
            print("sentence_level_span error!")
    return sentence_level_annotation_span_list

def transer(sentence):
    sentence = re.sub("\(", "\\(", sentence)
    sentence = re.sub("\)", "\\)", sentence)
    sentence = re.sub("\?", "\\?", sentence)
    sentence = re.sub("\+", "\\+", sentence)
    sentence = re.sub("\$", "\\$", sentence)
    sentence = re.sub("\[", "\\[", sentence)
    sentence = re.sub("\]", "\\]", sentence)
    sentence = re.sub("\*", "\\*", sentence)
    sentence = re.sub("\.", "\\.", sentence)
    return sentence


def index_blank(item, sentence, old_sentence):
    # item_processed, item_index = item
    # span_list = list(re.finditer(transer(item_processed), sentence))
    # if len(span_list) > 1:
    #     raise NameError("Find mutil index!")
    # elif len(span_list) == 1:
    #     insert_blank_index = re.search(" ", item_processed).span()[0] + span_list[0].span()[0]
    # return [insert_blank_index]
    item_processed, item_index = item
    insert_blank_index_list = []
    span_list = list(re.finditer(transer(item_processed), sentence))

    for i in span_list:
        insert_blank_index = re.search(" ", item_processed).span()[0] + i.span()[0]
        insert_blank_index_list.append(insert_blank_index)

    return insert_blank_index_list


def clean_sen(old_sentence):
    sentence = old_sentence.replace("\n", " ")

    insert_blank_index_and_item_list = []

    # front_parenthness_reg = "[\w]+\("
    front_parenthness_reg = "[^ ]+\([^ ^)]*"
    for i in re.finditer(front_parenthness_reg, sentence):
        item = i.group(0)
        item_processed = item.split("(")[0] + " (" + item.split("(")[1]
        sentence = re.sub(transer(item), item_processed, sentence)
        insert_blank_index_and_item_list.append((item_processed, i.span()))


    back_parenthness_reg = "[^ ]*\)[\w]+"
    for i in re.finditer(back_parenthness_reg, sentence):
        item = i.group(0)
        item_processed = item.split(")")[0] + ") " + item.split(")")[1]
        sentence = re.sub(transer(item), item_processed, sentence)
        insert_blank_index_and_item_list.append((item_processed, i.span()))

    comma_reg = "[\w),]+,[^ ^,]"
    for i in re.finditer(comma_reg, sentence):
        item = i.group(0)
        item_processed = re.sub(",", ", ", item)
        sentence = re.sub(transer(item), item_processed, sentence)
        insert_blank_index_and_item_list.append((item_processed, i.span()))

    insert_blank_list = []
    for i in insert_blank_index_and_item_list:
        insert_blank_index = index_blank(i, sentence, old_sentence)
        insert_blank_list += insert_blank_index

    return sentence, insert_blank_list


def new_sentence_span(raw_sentence_level_span_list, blank_span):
    if len(blank_span) > 0:
        new_sentence_span_list = []
        key_list = []
        span_list = []
        name_propagation_list = []
        parnessthess_list = []

        for key, annotation_span, name_propagation, parnessthess in raw_sentence_level_span_list:
            key_list.append(key)
            span_list.append((int(annotation_span.split("~")[0]), int(annotation_span.split("~")[1])))
            name_propagation_list.append(name_propagation)
            parnessthess_list.append(parnessthess)

        for blank_index in blank_span:
            for i in range(len(key_list)):
                raw_start = span_list[i][0]
                raw_end = span_list[i][1]

                new_tuple = []
                if int(blank_index) <= int(raw_start):
                    new_tuple.append(int(raw_start) + 1)
                else:
                    new_tuple.append(int(raw_start))

                if int(blank_index) <= int(raw_end):
                    new_tuple.append(int(raw_end) + 1)
                else:
                    new_tuple.append(int(raw_end))

                span_list[i] = (new_tuple[0], new_tuple[1])

                # new_sentence_span_list.append((key, str(temp_start) + "~" + str(temp_end)))

        for i in range(len(key_list)):
            new_sentence_span_list.append((key_list[i], str(span_list[i][0])+"~"+str(span_list[i][1]), name_propagation_list[i], parnessthess_list[i]))

        # for blank_index in blank_span:
        #     for key, annotation_span in raw_sentence_level_span_list:
        #         raw_start, raw_end = annotation_span.split("~")
        #         if int(blank_index) <= int(raw_start):
        #             temp_start = int(raw_start) + 1
        #         else:
        #             temp_start = int(raw_start)
        #
        #         if int(blank_index) <= int(raw_end):
        #             temp_end = int(raw_end) + 1
        #         else:
        #             temp_end = int(raw_end)
        #
        #     new_sentence_span_list.append((key, str(temp_start)+"~"+str(temp_end)))
        return new_sentence_span_list
    else:
        return raw_sentence_level_span_list


def check_new_sen_anno(new_sentence_span_list, new_sentence, sentence, raw_sentence_level_span_list):
    try:
        assert len(new_sentence_span_list) == len(raw_sentence_level_span_list)
    except:
        print("error!!!!!!")
        print(sentence.attrib["sentence_text"])
        raise NameError("new_sentence_span_list and raw_sen have different length!!!")

    for i in range(len(new_sentence_span_list)):
        old_start, old_end = raw_sentence_level_span_list[i][1].split("~")
        new_start, new_end = new_sentence_span_list[i][1].split("~")
        cleaned_sen, _ = clean_sen(sentence.attrib["sentence_text"][int(old_start): int(old_end)])
        if cleaned_sen != new_sentence[int(new_start): int(new_end)].strip():
            raise NameError("Wrong modified sentence!!!!!!!!!!!!!!!!!")

def generated_writen_str(annotation_span_list_type, sen_span_type, blank_span, writen_str, location_flag=False, relation_flag=False):
    if len(annotation_span_list_type) >= 1:
        raw_sentence_level_span_list = sentence_level_span(annotation_span_list_type, sen_span_type)  # 调整位置
        new_sentence_span_list = new_sentence_span(raw_sentence_level_span_list, blank_span)
        if relation_flag:
            writen_str = writen_str + "|" + str(new_sentence_span_list)
        elif location_flag:
            location_special = []
            for num in range(len(new_sentence_span_list)):
                location_special.append(new_sentence_span_list[num][:2])
            writen_str = writen_str + "|" + str(location_special)
        else:
            writen_str = writen_str + "|" + str(new_sentence_span_list[0][:2])

# check_new_sen_anno(new_sentence_span_list, new_sentence, sentence, raw_sentence_level_span_list)
    else:
        writen_str = writen_str + "|"

    return writen_str


if __name__ == "__main__":
    dir_source = '../sentence_level_corpus_add_death'
    dir_target = '../sentence_level_corpus_all_information.csv'
    # with open(dir_target, 'w') as f:
    #     f.write("Obituary_ID|Sentence|Death_Name|Death_Date|Birth_Date|Age_ID|Location_ID|relationship\n")

    file_count = 0
    sent_count = 0
    for tuple in os.walk(dir_source):
        for file in tuple[2]:
            global Deathname
            Deathname = ""
            file = os.path.join(dir_source, file)

            # file = os.path.join(dir_source, '1749.txt.xml')
            # print(file_count)
            # print(file)

            file_count += 1
            doc = etree.parse(file)
            sentence_list = doc.xpath("//sentence")
            sent_count += len(sentence_list)

            location_Name_list = doc.xpath("//Location-Name")

            NameText_locationID_dic = {}
            existing_name_list = []
            existing_location_list = []
            for location_Name in location_Name_list:
                NameText_locationID_dic[location_Name.attrib["LocationID"]] = location_Name.attrib["NameText"]
                if location_Name.attrib["NameID"] not in existing_name_list:
                    existing_name_list.append(location_Name.attrib["NameID"])
                # else:
                #     print()
                existing_location_list.append(location_Name.attrib["LocationID"])

            for sentence in sentence_list:
                annotation_span_list = []
                annotation_span_list_relationship = [] # 亲属
                annotation_span_list_name = []  # 姓名
                annotation_span_list_Death_Date = []  # 死亡日期
                annotation_span_list_age = []   # 死亡年龄
                annotation_span_list_Birth_Date = [] # 出生年龄
                annotation_span_list_Location = []  # 地点
                sen_span_name = ""
                sen_span_relationship = ""
                sen_span_Death_Date = ""
                sen_span_age = ""
                sen_span_Birth_Date = ""
                sen_span_Location = ""
                death_name = []

                for child in sentence.iterchildren():
                    # we can't recognize the deceased well, but just do it here
                    if "Name_ID" in child.attrib['id']:
                        annotation_span_list_name.append((child.attrib["Type"], child.attrib["spans"]))
                        sen_span_name = sentence.attrib["sentence_span"]
                        annotation_span_list.append(len(annotation_span_list_name))
                        death_name.append(child.attrib["text"])
                        Deathname = child.attrib["text"]

                    if "Death_Date_ID" in child.attrib['id']:
                        annotation_span_list_Death_Date.append(("Death_Date", child.attrib["spans"]))
                        sen_span_Death_Date = sentence.attrib["sentence_span"]
                        annotation_span_list.append(len(annotation_span_list_Death_Date))

                    if "Birth_Date_ID" in child.attrib['id']:
                        annotation_span_list_Birth_Date.append(("Birth_Date", child.attrib["spans"]))
                        sen_span_Birth_Date = sentence.attrib["sentence_span"]
                        annotation_span_list.append(len(annotation_span_list_Birth_Date))

                    if "Age_ID" in child.attrib['id']:
                        annotation_span_list_age.append(("Age", child.attrib["spans"]))
                        sen_span_age = sentence.attrib["sentence_span"]
                        annotation_span_list.append(len(annotation_span_list_age))

                    if "Location_ID" in child.attrib['id']:
                        Location_text = child.attrib["text"]
                        Location_span = child.attrib["spans"]


                        if child.attrib['id'] not in NameText_locationID_dic:
                            # annotation_span_list_Location.append(("Location-Death", Location_span))
                            annotation_span_list_Location.append(("Location-" + Deathname, Location_span))
                        else:
                            try:
                                if NameText_locationID_dic[child.attrib['id']] not in death_name:
                                    Location_name_value = NameText_locationID_dic[child.attrib['id']]
                                    annotation_span_list_Location.append(("Location-" + Location_name_value, Location_span))
                                else:
                                    # annotation_span_list_Location.append(("Location-Death", Location_span))
                                    annotation_span_list_Location.append(("Location-" + Deathname, Location_span))
                            except:
                                print("error")
                                print(child.attrib['id'])
                                print(NameText_locationID_dic)
                                print(file)


                        sen_span_Location = sentence.attrib["sentence_span"]
                        annotation_span_list.append(len(annotation_span_list_Location))

                    if ("G0_ID" in child.attrib['id']) or ("G1_ID" in child.attrib['id']) or ("G2_ID" in child.attrib['id'])  \
                            or ("G3_ID" in child.attrib['id']) or ("G4_ID" in child.attrib['id']) \
                                or ("G_minus_1" in child.attrib['id']) or ("G_minus_2" in child.attrib['id']) \
                                    or ("Other" in child.attrib['id']):

                        # relation pair format:(relation, span, name-propagation, parentheses)
                        if "Other" in child.attrib['id']:
                            type = "Other"
                        else:
                            type = child.attrib['type']

                        if child.attrib.__contains__("parentheses") and child.attrib.__contains__("name-propagation"):
                            annotation_span_list_relationship.append((type, child.attrib["spans"], child.attrib["name-propagation"], child.attrib["parentheses"]))
                        elif child.attrib.__contains__("name-propagation"):
                            annotation_span_list_relationship.append((type, child.attrib["spans"], child.attrib["name-propagation"], "None"))
                        else:
                            annotation_span_list_relationship.append((type, child.attrib["spans"], "no", "None"))

                        sen_span_relationship = sentence.attrib["sentence_span"]
                        annotation_span_list.append(len(annotation_span_list_relationship))

                # if sentence.attrib["sentence_text"] =="She is survived by a sister, Rose Frank, Sacramento, California, a son, James Frank (Barbara), Ellonton, FL, a daughter, Mary Kay Tointon (Glenn), N. Ft. Myers, Fl. and DePere, WI., daughter-in-law, Elaine Frank, San Antonio, TX, step daughter, Idelle Manthei(Terry) of Rochester, Mn., grandchildren, Tim (Tracy), Tonya, Christopher (Becca), James Jr. (Barbara), Thomas(Jamie), Steven(Lena), Kimberly, Heather, Robert Jr., Jessica(Mike),Craig and 20 great grandchildren.":
                #     print(1)
                if annotation_span_list != []:
                    writen_str = str(os.path.basename(file)).split(".")[0]
                    new_sentence, blank_span = clean_sen(sentence.attrib["sentence_text"])  # 正则化
                    writen_str = writen_str + "|" + new_sentence

                    writen_str = generated_writen_str(annotation_span_list_name, sen_span_name, blank_span, writen_str)
                    writen_str = generated_writen_str(annotation_span_list_Death_Date, sen_span_Death_Date, blank_span, writen_str)
                    writen_str = generated_writen_str(annotation_span_list_Birth_Date, sen_span_Birth_Date, blank_span, writen_str)
                    writen_str = generated_writen_str(annotation_span_list_age, sen_span_age, blank_span, writen_str)
                    writen_str = generated_writen_str(annotation_span_list_Location, sen_span_Location, blank_span, writen_str, location_flag=True)
                    writen_str = generated_writen_str(annotation_span_list_relationship, sen_span_relationship, blank_span, writen_str, relation_flag=True)

                    with open(dir_target, 'a', encoding="utf-8") as f:
                        f.write(writen_str + "\n")

    print(sent_count)