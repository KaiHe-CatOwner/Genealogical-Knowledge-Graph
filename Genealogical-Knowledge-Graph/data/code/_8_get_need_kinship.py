kinship_list = [('grand aunt', 0), ('great-great grand-nephew', 0), ('great-great grand-niece', 0),
                ('grandparent-in-law', 0), ('grandfather-in-law', 0), ('great-great grandchild-in-law', 0),

                ('cousin-in-law', 1), ('grandmother-in-law', 1), ('great-great grandson', 1),
                ('great-great granddaughter', 1),
                ('grand uncle', 1), ('great grand-nephew', 2), ('step-parent', 2), ('uncle-in-law', 3),
                ('great grandchild-in-law', 4),
                ('aunt-in-law', 6), ('step-granddaughter', 6), ('great grand-niece', 6), ('half-sister', 7),
                ('step-grandson', 7),

                ('grand child-in-law', 11), ('granddaughter-in-law', 12), ('grandson-in-law', 13), ('half-brother', 13),
                ('step-mother', 16),
                ('spouse', 18), ('ex-husband', 18), ('niece-in-law', 20), ('grand niece', 24), ('step-father', 24),
                ('grand nephew', 24),
                ('nephew-in-law', 25), ('child-in-law', 25), ('father-in-law', 26), ('great-great grand-child', 27),
                ('sibling-in-law', 28),
                ('grandfather', 29), ('mother-in-law', 30), ('ex-wife', 32), ('parent-in-law', 43), ('grandmother', 44),
                ('great granddaughter', 46),
                ('aunt', 49),

                ('uncle', 54), ('step-daughter', 60), ('great grandson', 65), ('step-son', 65), ('cousin', 91),
                ('step-grandchild', 98),
                ('son-in-law', 103), ('daughter-in-law', 114), ('son of', 132), ('father', 139), ('mother', 155),
                ('daughter of', 172), ('step-child', 175),
                ('grandparent', 210), ('granddaughter', 231), ('niece', 242), ('brother-in-law', 251), ('nephew', 297),
                ('grandson', 310), ('sister-in-law', 344),
                ('husband', 586), ('wife', 690), ('sibling', 718), ('parent', 720), ('other', 987),
                ('great grand-child', 1293), ('daughter', 1445), ('married to', 1457),
                ('son', 1713), ('brother', 2106), ('sister', 2156), ('born to', 2332), ('child', 2658),
                ('grand child', 4413)]


def return_need_kinship(strict_num):
    need_list = []
    dic_return = {}
    for relation , num in kinship_list:
        if num > strict_num:
            need_list.append(relation.replace(" ", "_"))

    for i in need_list:
        dic_return[i] = (0, 0, 0)

    return dic_return


if __name__ == "__main__":
    print(len(return_need_kinship(0)))
    print(len(kinship_list))