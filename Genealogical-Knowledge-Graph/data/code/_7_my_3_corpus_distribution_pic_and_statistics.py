import lxml.etree as etree
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter



need_num = 200

def parse_xml(file):
    doc = etree.parse(file)
    text = doc.xpath("//TEXT")[0].text
    name_list = doc.xpath("//TAGS/Name")
    age_list = doc.xpath("//TAGS/Age")
    location_list = doc.xpath("//TAGS/Location")
    death_date_list = doc.xpath("//TAGS/Death_Date")
    dirth_date_list = doc.xpath("//TAGS/Birth_Date")
    Location_Name_Link_list = doc.xpath("//TAGS/Location-Name")
    G0_list = doc.xpath("//TAGS/G_0")
    G1_list = doc.xpath("//TAGS/G_1")
    G2_list = doc.xpath("//TAGS/G_2")
    G3_list = doc.xpath("//TAGS/G_3")
    G4_list = doc.xpath("//TAGS/G_4")
    Gm1_list = doc.xpath("//TAGS/G_minus_1")
    Gm2_list = doc.xpath("//TAGS/G_minus_2")
    other_list = doc.xpath("//TAGS/Other")

    return text, name_list, age_list, location_list, death_date_list, dirth_date_list, \
            G0_list, G1_list, G2_list, G3_list, G4_list, \
                Gm1_list, Gm2_list, other_list, Location_Name_Link_list


def count_parentheses_count(relation_lists):
    temp_count = 0
    for relation_list in relation_lists:
        for item in relation_list:
            try:
                if item.attrib['parentheses']:
                    temp_count += 1
            except:
                pass

    return temp_count


def count_surname_delivery(relation_lists):
    temp_count = 0
    for relation_list in relation_lists:
        for item in relation_list:
            try:
                if item.attrib['name-propagation'] == 'yes':
                    temp_count += 1
            except:
                pass

    return temp_count


def count_relation(relation_lists, dic_kin_count):
    temp_count = 0
    for relation_list in relation_lists:
        for item in relation_list:
            try:
                if item.attrib['type']:
                    temp_count += 1
                    dic_kin_count[item.attrib['type']] += 1
            except:
                if "Other" not in item.attrib['id']:
                    print("there is no type error!")
                else:
                    dic_kin_count["other"] += 1
                    temp_count += 1
                pass

    return temp_count, dic_kin_count


def count_parentheses_content(relation_lists):
    nick_name_count = 0
    middle_name_count = 0
    spouse_name_count = 0
    previous_name_count = 0
    for relation_list in relation_lists:
        for item in relation_list:
            try:
                if item.attrib['parentheses'] == "nickname":
                    nick_name_count += 1
                if item.attrib['parentheses'] == "middlename":
                    middle_name_count += 1
                if item.attrib['parentheses'] == "coupled-name":
                    spouse_name_count += 1
                if item.attrib['parentheses'] == "previous-last-name":
                    previous_name_count += 1
            except:
                pass
    return nick_name_count, middle_name_count, spouse_name_count, previous_name_count


def get_result():
    return sorted_dic_kin_count


def main():
    dir_0 = "../processed_data_Location_ID_corrected"
    file_list = []
    for file_tuple in os.walk(dir_0):
        for file in file_tuple[2]:
            file = os.path.join(dir_0, file)
            file_list.append(file)

    print("There are %s file !" % str(len(file_list)))

    name_count = 0
    death_name_count = 0
    parentheses_count = 0
    surname_delivery_count = 0

    relation_count = 0
    location_count = 0
    Location_name_link_count = 0

    nick_name_count = 0
    middle_name_count = 0
    spouse_name_count = 0
    previous_name_count = 0

    age_count = 0
    death_date_count = 0
    birth_date_count = 0

    dic_kin_count = {}
    kin_tpye_list = ["ex-husband","ex-wife","married to","spouse","husband","wife","sibling","cousin","brother","sister","half-brother","half-sister","sister-in-law","sibling-in-law","cousin-in-law","brother-in-law",
                    "child","daughter","son","niece","nephew","step-child","step-daughter","step-son","child-in-law","daughter-in-law","son-in-law","niece-in-law","nephew-in-law",
                    "grandson","grand child","granddaughter","grand nephew","grand niece","grandson-in-law","grand child-in-law","granddaughter-in-law","step-grandchild","step-grandson","step-granddaughter",
                    "great grand-child","great granddaughter","great grandson","great grand-nephew","great grand-niece","great grandchild-in-law",
                    "great-great grand-child","great-great granddaughter","great-great grandson","great-great grand-nephew","great-great grand-niece","great-great grandchild-in-law",
                    "born to","son of","daughter of","parent","mother","father","step-mother","step-father","step-parent","aunt","uncle","parent-in-law","mother-in-law","father-in-law","aunt-in-law","uncle-in-law",
                    "grandparent","grandmother","grandfather","grand aunt","grand uncle","grandparent-in-law","grandmother-in-law","grandfather-in-law",
                     "other"
                     ]

    temp_id = 0
    for i in kin_tpye_list:
        dic_kin_count[i] = 0

    # file_list = ['./processed_data_old/801.txt.xml']
    # print("test_set++++++++++++++++++++==='", file_list)
    # print()

    G0 = 0
    G1 = 0
    G2 = 0
    G3 = 0
    G4 = 0
    Gm1 = 0
    Gm2 = 0
    G_other = 0
    for file in file_list:
        text, name_list, age_list, location_list, death_date_list, dirth_date_list, \
        G0_list, G1_list, G2_list, G3_list, G4_list, \
        Gm1_list, Gm2_list, other_list, Location_Name_Link_list = parse_xml(file)

        death_name_count += len(name_list)
        name_count += len(name_list)
        name_count += len(G0_list)
        name_count += len(G1_list)
        name_count += len(G2_list)
        name_count += len(G3_list)
        name_count += len(G4_list)
        name_count += len(Gm1_list)
        name_count += len(Gm2_list)
        name_count += len(other_list)

        G0+=len(G0_list)
        G1+=len(G1_list)
        G2+=len(G2_list)
        G3+=len(G3_list)
        G4+=len(G4_list)
        Gm1+=len(Gm1_list)
        Gm2+=len(Gm2_list)
        G_other+=len(other_list)

        parentheses_count += count_parentheses_count((G0_list, G1_list, G2_list, G3_list, G4_list, Gm1_list, Gm2_list, other_list))
        surname_delivery_count += count_surname_delivery((G0_list, G1_list, G2_list, G3_list, G4_list, Gm1_list, Gm2_list, other_list))
        relations_return = count_relation((G0_list, G1_list, G2_list, G3_list, G4_list, Gm1_list, Gm2_list, other_list), dic_kin_count)
        relation_count += relations_return[0]
        dic_kin_count = relations_return[1]

        location_count += len(location_list)
        Location_name_link_count += len(Location_Name_Link_list)
        parentheses_content = count_parentheses_content((G0_list, G1_list, G2_list, G3_list, G4_list, Gm1_list, Gm2_list, other_list))
        nick_name_count += parentheses_content[0]
        middle_name_count += parentheses_content[1]
        spouse_name_count += parentheses_content[2]
        previous_name_count += parentheses_content[3]

        # if parentheses_content[3]>0:
        #     print(text)

        age_count += len(age_list)
        death_date_count += len(death_date_list)
        birth_date_count += len(dirth_date_list)

        if len(death_date_list)>1:
            print("*****************************************")
        if len(dirth_date_list)>1:
            print("-------------------------------------")

    print("G0", G0)
    print("G1", G1)
    print("G2", G2)
    print("G3", G3)
    print("G4", G4)
    print("Gm1", Gm1)
    print("Gm2", Gm2)
    print("G_other", G_other)


    if parentheses_count != (nick_name_count+middle_name_count+spouse_name_count+previous_name_count):
        print("parentheses_count errors!===========")

    print("name_count", name_count)
    print("death_name_count", death_name_count)

    print("surname_delivery_count", surname_delivery_count)
    print("relation_count", relation_count)

    print("location_count", location_count)
    print("Location_name_link_count", Location_name_link_count)

    print("parentheses_count", parentheses_count)
    print("nick_name_count", nick_name_count)
    print("middle_name_count", middle_name_count)
    print("spouse_name_count", spouse_name_count)
    print("previous_name_count", previous_name_count)
    sorted_dic_kin_count = sorted(dic_kin_count.items(), key=lambda x: x[1])

    #dic counting==============================================
    print(len(sorted_dic_kin_count))
    print(sorted_dic_kin_count)
    count_0 = 0
    count_50 = 0
    count_100 = 0
    count_more_100 = 0
    for item in sorted_dic_kin_count:
        count = int(item[1])
        if count ==0:
            count_0 += 1

        if count > 0 and count <51:
            count_50 += 1

        if count > 50 and count <101:
            count_100 += 1

        if count > 100 :
            count_more_100 += 1

    print("count_0 : ", count_0)
    print("count_50 : ", count_50)
    print("count_100 : ", count_100)
    print("count_more_100 : ", count_more_100)

    total = 0
    for item,value in dic_kin_count.items():
        total += value

    print('counting total relation :', total)

    print("age_count", age_count)
    print("death_date_count", death_date_count)
    print("birth_date_count", birth_date_count)

    G0_count_list = []
    G1_count_list = []
    G2_count_list = []
    G3_count_list = []
    G4_count_list = []
    Gm1_count_list = []
    Gm2_count_list = []
    other_count_list = []

    G0_list= ["ex-husband","wife","ex-wife","married to","husband","spouse","cousin", "sibling","brother","half-brother","half-sister","sister","sister-in-law","sibling-in-law","cousin-in-law","brother-in-law"]
    G1_list= ["child","daughter","nephew","step-child","step-daughter","niece","step-son","child-in-law","son","daughter-in-law","son-in-law","niece-in-law","nephew-in-law"]
    G2_list= ["grand child","grandson","grandson-in-law", "grand child-in-law","granddaughter","grand nephew","grand niece","granddaughter-in-law","step-grandchild","step-grandson","step-granddaughter"]
    G3_list= ["great grand-child","great granddaughter","great grandson","great grand-nephew","great grand-niece","great grandchild-in-law"]
    G4_list= ["great-great grand-child","great-great granddaughter","great-great grandson","great-great grand-nephew","great-great grand-niece","great-great grandchild-in-law"]
    Gm1_list= ["born to","son of","daughter of","parent","mother","father","step-mother","step-father","step-parent","aunt","uncle","parent-in-law","mother-in-law","father-in-law","aunt-in-law","uncle-in-law"]
    Gm2_list= ["grandparent","grandmother","grandfather","grand aunt","grand uncle","grandparent-in-law","grandmother-in-law","grandfather-in-law"]
    other_list= ["other"]
    color_list = ["black", "darkorange", "limegreen", "red", "blue", "deeppink", "grey", "aqua", "darkgreen", "yellow"]

    kin_type_count_list_total = [ (other_list, other_count_list), (G0_list, G0_count_list), (G1_list, G1_count_list), (G2_list, G2_count_list),
                                 (G3_list, G3_count_list), (G4_list, G4_count_list), (Gm1_list, Gm1_count_list),
                                 (Gm2_list,Gm2_count_list )]


    for kin_type_count_list in kin_type_count_list_total:
        for kin in kin_type_count_list[0]:
            kin_type_count_list[1].append(dic_kin_count[kin])




    # tick_label =  ["G0_list", "G1_list", "G2_list", "G3_list", "G4_list", "Gm1_list", "Gm2_list", "other_list"]
    #
    # c_index = 0
    # temp_count = 0
    # yticks_list = [0]
    # for i in kin_type_count_list_total:
    #     id_list = [i+temp_count for i in range(len(i[0]))]
    #     temp_bars = plt.barh(id_list,  i[1], color=color_list[c_index], height=0.8)
    #
    #     for kin_index in range(len(i[1])):
    #         if i[1][kin_index]> need_num:
    #             plt.text(i[1][kin_index] + 30, id_list[kin_index], '%s' % i[0][kin_index], ha='left', va='center', fontsize=7)
    #
    #     c_index +=1
    #     temp_count += len(i[0])
    #     # temp_count += 2
    #     yticks_list.append(temp_count)
    #
    # plt.xticks([0, 500, 1000,1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500])
    #
    # scale_ls_te = yticks_list
    # scale_ls_te = [i-0.5 for i in scale_ls_te]
    # index_ls_te = []
    # _ = plt.yticks(scale_ls_te, index_ls_te)
    #
    # x = -450
    # plt.text(x, (yticks_list[0]+yticks_list[1])/2-0.5, 'other')
    # plt.text(x+45, (yticks_list[1]+yticks_list[2])/2-0.5, 'G0')
    # plt.text(x+45, (yticks_list[2]+yticks_list[3])/2-0.5, 'G1')
    # plt.text(x+45, (yticks_list[3]+yticks_list[4])/2-0.5, 'G2')
    # plt.text(x+45, (yticks_list[4]+yticks_list[5])/2-0.5, 'G3')
    # plt.text(x+45, (yticks_list[5]+yticks_list[6])/2-0.5, 'G4')
    # plt.text(x, (yticks_list[6]+yticks_list[7])/2-0.5, 'Gm1')
    # plt.text(x, (yticks_list[7]+yticks_list[7])/2+3, 'Gm2')

    # for bar in temp_bars:
    #     w = bar.get_width()
    #     if w > 300:
    #         plt.text(w, bar.get_y() + bar.get_height() / 2, '%s' % i[0][temp_bar_index], ha='left', va='center')


    # label_list = ["G0", "G1", "G2", "G3", "G4", "Gm1", "Gm2", "other"]
    # label_list_count = 0
    # for i in kin_type_count_list_total:
    #     plt.barh(i[0], i[1], label=label_list[label_list_count])
    #     label_list_count +=1
    #
    # y = -165
    # plt.text(6, y, 'G0')
    # plt.text(21, y, 'G1')
    # plt.text(33, y, 'G2')
    # plt.text(42, y, 'G3')
    # plt.text(47, y, 'G4')
    # plt.text(59, y, 'Gm1')
    # plt.text(69, y, 'Gm2')
    # plt.text(76, y, 'other')
    #
    # plt.legend()
    # plt.xticks([])
    #
    # frame = plt.gca()
    # frame.axes.get_xaxis().set_visible(False)
    #
    #
    ## 可以设置坐标字
    # scale_ls_te = [0, 16, 29, 40, 46, 52, 68, 76]
    # scale_ls_te = [i-0.5 for i in scale_ls_te]
    # index_ls_te = []
    # _ = plt.xticks(scale_ls_te, index_ls_te)
    # plt.ylabel(u"count")

    # plt.subplots_adjust(left = None, bottom = None, right = None, top = None)
    # plt.savefig("./F2-300.png", dpi=350, pad_inches=0)
    # plt.show()

    return sorted_dic_kin_count



sorted_dic_kin_count = main()


