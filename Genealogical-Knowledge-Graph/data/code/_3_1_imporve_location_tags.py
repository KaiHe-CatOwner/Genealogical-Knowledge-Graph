def get_relationship_index(relatinship_list, sentence):
    relatinship_start_index = []
    relatinship_name_list = []
    for relationship in relatinship_list:
        start_index, end_index = relationship[1].split("~")
        relatinship_start_index.append(int(start_index))
        relatinship_name_list.append(sentence[int(start_index):int(end_index)])
    return relatinship_start_index, relatinship_name_list

# john and patricia of rochester  只标记了第一个pair(john, rochester) （写到预处理里 and形式）  ===> pair(john and patricia, rochester)  processed in model file
if __name__ == '__main__':
    midden_number = 30

    with open("../sentence_level_corpus_all_information_normalized.csv", "r") as f:
        raw_data = f.readlines()
    result_file = "../sentence_level_corpus_all_information_normalized_improve_location.csv"

    new_list = []
    for n, i in enumerate(raw_data[1:]):
        new_items = i
        items = i.split("|")
        # items = [Obituary_ID, Sentence, Death_Name, Death_Date, Birth_Date, Age_ID, Location_ID, relationship]
        # items_index  [0,          1,       2,            3,          4,         5,      6            7]
        if items[6] != "" and items[7] != "\n":
            relatinship_start_index, relatinship_name_list = get_relationship_index(eval(items[7]), items[1])
            Location_list = eval(items[6])
            for number, Location in enumerate(Location_list):
                if " and " in Location[0]:
                    continue
                name_index = items[1].find(Location[0].split("tion-")[1])
                Location_index = int(Location[1].split("~")[0])
                if Location_index - name_index < midden_number:
                    for start_index, name in zip(relatinship_start_index, relatinship_name_list):
                        if start_index > name_index and start_index < Location_index:
                            new_Location_str = Location[0]+ " and " + name
                            Location = (new_Location_str, Location[1])
                            Location_list[number] = Location
            new_items = str(items[0]) + "|" + str(items[1]) + "|" + str(items[2]) + "|" + str(
                items[3]) + "|" + str(items[4]) + "|" + str(items[5]) + "|" + str(Location_list) + "|" + str(items[7])
            new_list.append(new_items)
        else:
            new_list.append(new_items)

    with open(result_file, "w") as f:
        f.writelines("Obituary_ID|Sentence|Death_Name|Death_Date|Birth_Date|Age_ID|Location_ID|relationship\n")
        for i in new_list:
            f.writelines(i)
