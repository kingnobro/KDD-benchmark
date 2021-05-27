#!/usr/bin/env python
# encoding: utf-8
# import sys
# import importlib
# importlib.reload(sys)
# sys.setdefaultencoding('utf-8')
import util
import numpy as np
import pandas as pd
import textdistance
import nltk
import re

# 2. coauthor信息
# 很多论文都有多个作者，根据paperauthor统计每一个作者的top 10（当然可以是top 20或者其他top K）的coauthor，
# 对于一个作者论文对（aid，pid），计算ID为pid的论文的作者有没有出现ID为aid的作者的top 10 coauthor中，
# (1). 可以简单计算top 10 coauthor出现的个数，
# (2). 还可以算一个得分，每个出现pid论文的top 10 coauthor可以根据他们跟aid作者的合作次数算一个分数，然后累加，
# 我简单地把coauthor和当前aid作者和合作次数作为这个coauthor出现的得分。


# count of coauthors with the same affiliation
def affiliation_count(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Conference, Journal):
    authorId = AuthorIdPaperId.authorId
    paperId = AuthorIdPaperId.paperId
    # 从PaperAuthor中，根据paperId找coauthor
    curr_coauthors = list(map(str, PaperAuthor[PaperAuthor["PaperId"] == int(paperId)]["AuthorId"].values))
    curr_affiliations = list(map(str, PaperAuthor[PaperAuthor["PaperId"] == int(paperId)]["Affiliation"].values))
    index = 0
    for author in curr_coauthors:
        if author == authorId:
            break
        index += 1
    affiliation = curr_affiliations[index]
    if affiliation == 'nan':
        return util.get_feature_by_list([0])
    else:
        return util.get_feature_by_list([curr_affiliations.count(affiliation)])


def journal_count(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Conference, Journal):
    authorId = int(AuthorIdPaperId.authorId)
    paperIds = PaperAuthor[PaperAuthor['AuthorId'] == int(authorId)]['PaperId'].values
    journalIds = set()
    for id in paperIds:
        journalId = Paper[Paper['Id'] == int(id)]['JournalId'].values
        if len(journalId) > 0:
            journalIds.add(int(journalId[0]))

    return util.get_feature_by_list([len(journalIds)])


def journal_conference_year(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Conference, Journal):
    paperId = int(AuthorIdPaperId.paperId)

    conferenceId = Paper[Paper['Id'] == paperId]['ConferenceId'].values
    journalId = Paper[Paper['Id'] == paperId]['JournalId'].values
    paper_year = int(Paper[Paper['Id'] == int(paperId)]['Year'].values[0])

    feat_list = []
    if len(conferenceId) == 0 or int(conferenceId[0]) <= 0:
        feat_list.append(0)
    else:
        feat_list.append(int(conferenceId[0]))

    if len(journalId) == 0 or int(journalId[0]) <= 0:
        feat_list.append(0)
    else:
        feat_list.append(int(journalId[0]))

    if 1800 <= paper_year <= 2013:
        feat_list.append(paper_year)
    else:
        feat_list.append(0)


    return util.get_feature_by_list(feat_list)


def keyword(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Conference, Journal):
    def get_words(paper):
        s = str(paper.Title)
        if not pd.isna(paper.Keyword):
            s += paper.Keyword
        # print(s)
        words = re.split(r'[|\s;,]', s)
        words = [w for w in words if w and w not in nltk.corpus.stopwords.words('english') and not w.isdigit()]
        return words

    authorId = AuthorIdPaperId.authorId
    paperId = AuthorIdPaperId.paperId

    papersOfAuthor = PaperAuthor[PaperAuthor['AuthorId'] == int(authorId)]
    kws = get_words(Paper[Paper['Id'] == int(paperId)].iloc[0])

    feature = []
    if papersOfAuthor.shape[0] == 0:
        feature += [0]
    else:
        cnt = 0
        s = set()
        for _, row in papersOfAuthor.iterrows():
            paper = Paper[Paper['Id'] == row.PaperId]
            if paper.shape[0] == 0:
                continue
            paper = paper.iloc[0]
            _kws = get_words(paper)
            cnt += len(_kws)
            if paper.Id != paperId:
                s.update(_kws)
        feature.append(len(s.intersection(set(kws))))

    return util.get_feature_by_list(feature)


def publication_year(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Conference, Journal):
    authorId = AuthorIdPaperId.authorId
    paperId = AuthorIdPaperId.paperId
    # print('authorId', authorId)

    # paperId 的发表年份
    paper_year = Paper[Paper['Id'] == int(paperId)]['Year'].values[0]

    # 作者发表的所有论文 id
    paperIds = PaperAuthor[PaperAuthor['AuthorId'] == int(authorId)]['PaperId'].values
    years = []
    for id in paperIds:
        year = Paper[Paper['Id'] == int(id)]['Year'].values
        if year.shape[0] == 0:
            continue
        year = year[0]
        if 1800 <= year <= 2013:
            years.append(year)

    if not years:
        feature = [0, 0, 0]
    else:
        feature = [1, paper_year - min(years), max(years) - paper_year]

    return util.get_feature_by_list(feature)


# 1. 简单计算top 10 coauthor出现的个数
def coauthor_1(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Conference, Journal):
    authorId = AuthorIdPaperId.authorId  # int
    paperId = AuthorIdPaperId.paperId  # int

    # 从PaperAuthor中，根据paperId找coauthor。
    curr_coauthors = list(map(str, list(PaperAuthor[PaperAuthor["PaperId"] == int(paperId)]["AuthorId"].values)))
    #
    top_coauthors = list(dict_coauthor[authorId].keys())

    # 简单计算top 10 coauthor出现的个数
    nums = len(set(curr_coauthors) & set(top_coauthors))

    return util.get_feature_by_list([nums])


# 2. 还可以算一个得分，每个出现pid论文的top 10 coauthor可以根据他们跟aid作者的合作次数算一个分数，然后累加，
def coauthor_2(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Conference, Journal):
    authorId = AuthorIdPaperId.authorId
    paperId = AuthorIdPaperId.paperId

    # 从PaperAuthor中，根据paperId找coauthor。
    curr_coauthors = list(map(str, list(PaperAuthor[PaperAuthor["PaperId"] == int(paperId)]["AuthorId"].values)))

    # {"authorId": 100}
    top_coauthors = dict_coauthor[authorId]

    score = 0
    for curr_coauthor in curr_coauthors:
        if curr_coauthor in top_coauthors:
            score += top_coauthors[curr_coauthor]

    return util.get_feature_by_list([score])


''' String Distance Feature'''


# 1. name-a 与name1##name2##name3的距离，同理affliction-a 和 aff1##aff2##aff3的距离
def stringDistance_1(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Conference, Journal):
    authorId = AuthorIdPaperId.authorId
    paperId = AuthorIdPaperId.paperId

    key = "%s|%s" % (paperId, authorId)
    name = str(dict_paperIdAuthorId_to_name_aff[key]["name"])
    aff = str(dict_paperIdAuthorId_to_name_aff[key]["affiliation"])

    T = list(Author[Author["Id"] == int(authorId)].values)[0]
    a_name = str(T[1])
    a_aff = str(T[2])
    if a_name == "nan":
        a_name = ""
    if a_aff == "nan":
        a_aff = ""

    feat_list = []

    # 计算 a_name 与 name 的距离
    feat_list.append(len(longest_common_subsequence(a_name, name)))
    feat_list.append(len(longest_common_substring(a_name, name)))
    # feat_list.append(Levenshtein_distance(a_name, name))
    # feat_list.append(textdistance.Jaccard()(a_name, name))
    feat_list.append(textdistance.JaroWinkler()(a_name, name))

    # 计算 a_aff 与 aff 的距离
    feat_list.append(len(longest_common_subsequence(a_aff, aff)))
    feat_list.append(len(longest_common_substring(a_aff, aff)))
    # feat_list.append(Levenshtein_distance(a_aff, aff))
    # feat_list.append(textdistance.Jaccard()(a_aff, aff))
    feat_list.append(textdistance.JaroWinkler()(a_aff, aff))

    return util.get_feature_by_list(feat_list)


# 2. name-a分别与name1，name2，name3的距离，然后取平均，同理affliction-a和,aff1，aff2，aff3的平均距离
def stringDistance_2(AuthorIdPaperId, dict_coauthor, dict_paperIdAuthorId_to_name_aff, PaperAuthor, Author, Paper, Conference, Journal):
    authorId = AuthorIdPaperId.authorId
    paperId = AuthorIdPaperId.paperId

    key = "%s|%s" % (paperId, authorId)
    name = str(dict_paperIdAuthorId_to_name_aff[key]["name"])
    aff = str(dict_paperIdAuthorId_to_name_aff[key]["affiliation"])

    T = list(Author[Author["Id"] == int(authorId)].values)[0]
    a_name = str(T[1])
    a_aff = str(T[2])
    if a_name == "nan":
        a_name = ""
    if a_aff == "nan":
        a_aff = ""

    feat_list = []

    # 计算 a_name 与 name 的距离
    lcs_distance = []
    lss_distance = []
    lev_distance = []
    for _name in name.split("##"):
        lcs_distance.append(len(longest_common_subsequence(a_name, _name)))
        lss_distance.append(len(longest_common_substring(a_name, _name)))
        # 尝试不同的字符串相似度算法
        # lev_distance.append(Levenshtein_distance(a_name, _name))
        lev_distance.append(textdistance.JaroWinkler()(a_name, _name))
        # lev_distance.append(textdistance.Jaccard()(a_name, _name))

    feat_list += [np.mean(lcs_distance), np.mean(lss_distance), np.mean(lev_distance)]

    # 计算 a_aff 与 aff 的距离
    lcs_distance = []
    lss_distance = []
    lev_distance = []
    for _aff in aff.split("##"):
        lcs_distance.append(len(longest_common_subsequence(a_aff, _aff)))
        lss_distance.append(len(longest_common_substring(a_aff, _aff)))
        # 尝试不同的字符串相似度算法
        # lev_distance.append(Levenshtein_distance(a_aff, _aff))
        lev_distance.append(textdistance.JaroWinkler()(a_aff, _aff))
        # lev_distance.append(textdistance.Jaccard()(a_aff, _aff))

    feat_list += [np.mean(lcs_distance), np.mean(lss_distance), np.mean(lev_distance)]

    # # feat_list
    # feat_list = [feat_list[0],feat_list[1], feat_list[3],feat_list[4]]

    return util.get_feature_by_list(feat_list)


''' 一些距离计算方法 '''


# 最长公共子序列（LCS）, 获取是a, b的最长公共子序列
def longest_common_subsequence(a, b):
    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    # read the substring out from the matrix
    result = ""
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            assert a[x - 1] == b[y - 1]
            result = a[x - 1] + result
            x -= 1
            y -= 1
    return result


# 最长公共子串（LSS）
def longest_common_substring(a, b):
    m = [[0] * (1 + len(b)) for i in range(1 + len(a))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(a)):
        for y in range(1, 1 + len(b)):
            if a[x - 1] == b[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return a[x_longest - longest: x_longest]


# 编辑距离
def Levenshtein_distance(input_x, input_y):
    xlen = len(input_x) + 1  # 此处需要多开辟一个元素存储最后一轮的计算结果
    ylen = len(input_y) + 1

    dp = np.zeros(shape=(xlen, ylen), dtype=int)
    for i in range(0, xlen):
        dp[i][0] = i
    for j in range(0, ylen):
        dp[0][j] = j

    for i in range(1, xlen):
        for j in range(1, ylen):
            if input_x[i - 1] == input_y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[xlen - 1][ylen - 1]


if __name__ == '__main__':
    print(Levenshtein_distance("abc", "ab"))
