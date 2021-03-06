from sklearn.datasets import load_svmlight_file
from sklearn import svm
import pandas as pd
import textdistance
# import textdistance
import nltk
import ssl
import math


def download_stopwords():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('stopwords')


if __name__ == '__main__':
    # download_stopwords()
    # train = pd.read_csv('./data/dataset/train_set/train.csv')
    # valid = pd.read_csv('./data/dataset/valid_set/valid.csv')
    PaperAuthor = pd.read_csv('./data/dataset/PaperAuthor.csv')
    curr_coauthors = list(map(str, PaperAuthor[PaperAuthor["PaperId"] == int(1291787)]["AuthorId"].values))
    curr_affiliations = list(map(str, PaperAuthor[PaperAuthor["PaperId"] == int(1291787)]["Affiliation"].values))
    print(curr_affiliations[0] == 'nan')

    print(curr_coauthors)
    print(curr_affiliations)
    # Paper = pd.read_csv('./data/dataset/Paper.csv')
    # for index, line in Paper.iterrows():
    #     cid = int(line['ConferenceId'])
    #     jid = int(line['JournalId'])
    #     if cid != 0 and jid != 0:
    #         print(cid, jid)
    # cid = Paper[Paper['Id'] == 1291787]['ConferenceId'].values[0]
    # jid = Paper[Paper['Id'] == 1291787]['JournalId'].values[0]
    # print(cid, jid)
    # print(Paper[Paper['Id'] == 9].iloc[0])
    # author1 = set(train['AuthorId'])
    # author2 = set(valid['AuthorId'])
    # print(author1.intersection(author2))
    # paperId = p

    # paperIds = PaperAuthor[PaperAuthor['AuthorId'] == int(authorId)]['PaperId'].values
    # kws = get_words(Paper[Paper['Id'] == paperId].iloc[0])

    #
    # paperId = 1291787
    # year = paper[paper['Id'] == paperId]['Year'].values[0]
    # print(year)
    # 根据 authorId 找到发表过的所有文章的 paperId
    # authorId = 1140870
    # paperIds = paperIdautherId[paperIdautherId['AuthorId'] == int(authorId)]['PaperId'].values
    # paperYears = []
    # for id in paperIds:
    #     year = paper[paper['Id'] == id]['Year'].values[0]
    #     if 1800 <= year <= 2013:
    #         paperYears.append(year)

    # print(sum(paperYears) / len(paperYears))
