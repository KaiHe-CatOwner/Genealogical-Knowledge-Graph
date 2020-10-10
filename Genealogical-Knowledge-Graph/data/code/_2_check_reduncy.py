# 将语句变成字典

with open("../sentence_level_corpus_all_information.csv", "r") as f:
    raw_data = f.readlines()
print("raw data:", len(raw_data[1:]))


dic = {}
for i in raw_data[1:]:
    temp_list = i.split("|")
    if temp_list[1] in dic.keys():
        print(i)
    dic[temp_list[1]] = temp_list[2]

print(len(dic))