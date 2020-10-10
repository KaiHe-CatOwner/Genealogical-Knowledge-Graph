normalized_list = [["spouse","married to"],
                   ["great-grandchild","great grand-child"],
                   ["great-grandchild-in-law","great grandchild-in-law"],
                   ["great-granddaughter","great granddaughter"],
                   ["great-grandson","great grandson"],
                   ["great-grandnephew","great grand-nephew"],
                   ["great-grandniece","great grand-niece"],
                   ["great-great-grandchild","great-great grand-child"],
                   ["great-great-granddaughter","great-great granddaughter"],
                   ["great-great-grandson","great-great grandson"],
                   ["parent","born to","son of","daughter of"],
                   ["granduncle","grand uncle"],
                   ["grandaunt","grand aunt"],
                   ["grandchild","grand child"],
                   ['grandchild-in-law',  'grand child-in-law']
                   ]


D1 = [
    # ['great-great grand-nephew', 0], ['great-great grand-niece', 0],
    ['great-great grandchild-in-law', 0],
    ['grand aunt', 0],
    ['grandparent-in-law', 0], ['grandfather-in-law', 0], ['cousin-in-law', 1], ['great-great granddaughter', 1],
    ['great-great grandson', 1], ['uncle-in-law', 1], ['grand uncle', 1], ['great grand-nephew', 2], ['step-parent', 2],
    ['aunt-in-law', 2], ['grandmother-in-law', 2], ['step-granddaughter', 4], ['great grand-niece', 6], ['half-sister', 7],
    ['great grandchild-in-law', 7], ['step-grandson', 8], ['grandson-in-law', 10], ['grand child-in-law', 10], ['half-brother', 12],
    ['granddaughter-in-law', 14], ['step-mother', 14], ['nephew-in-law', 15], ['great-great grand-child', 15], ['niece-in-law', 16],
    ['ex-husband', 17], ['spouse', 17], ['grand niece', 21], ['sibling-in-law', 22], ['grand nephew', 22], ['father-in-law', 22],
    ['child-in-law', 25], ['step-father', 25], ['mother-in-law', 27], ['ex-wife', 32], ['grandfather', 33], ['grandmother', 41],
    ['parent-in-law', 45], ['step-daughter', 48], ['great granddaughter', 49], ['aunt', 51], ['step-son', 55], ['uncle', 56],
    ['great grandson', 58], ['cousin', 90], ['step-grandchild', 92], ['daughter-in-law', 99], ['son-in-law', 105], ['son of', 127],
    ['father', 139], ['daughter of', 148], ['step-child', 151], ['mother', 151], ['grandparent', 207], ['niece', 217],
    ['granddaughter', 224], ['brother-in-law', 235], ['nephew', 251], ['grandson', 309], ['sister-in-law', 318], ['husband', 531],
    ['wife', 666], ['parent', 687], ['sibling', 707],  ['great grand-child', 1135], ['married to', 1345],['other', 1036],
    ['daughter', 1399], ['son', 1584], ['brother', 2012], ['sister', 2016], ['born to', 2182], ['child', 2489], ['grand child', 4088]]


replace_list = []
dic_replace = {}
for i in normalized_list:
    for j in i:
        replace_list.append(j)
    for key in i:
        dic_replace.update({key:i[0]})

for i in D1:
    if i[0] in dic_replace.keys():
        i[0] = dic_replace[i[0]]

# print(len(D1))
# print(D1)
for i0, v0  in enumerate(D1):
    for i1, v1  in enumerate(D1):
        if i1 != i0:
            if v0[0]==v1[0]:
                v0[1] = v0[1]+v1[1]
                D1.remove(v1)

dic_res = {}
for i in D1:
    dic_res[i[0]]=i[1]


from sklearn.preprocessing import MinMaxScaler
import numpy as np

minmaxScaler = MinMaxScaler(feature_range=(10, 60))
list_weight1 = minmaxScaler.fit_transform(np.array(list(dic_res.values())).reshape(-1, 1))

for i, j in enumerate(dic_res.keys()):
    dic_res[j]=float(list_weight1[i])

print(dic_res)




