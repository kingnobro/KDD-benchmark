from sklearn.datasets import load_svmlight_file
from sklearn import svm
import pandas as pd
import textdistance
# import textdistance


if __name__ == '__main__':
    train = pd.read_csv('./data/dataset/train_set/train.csv')
    valid = pd.read_csv('./data/dataset/valid_set/valid.csv')
    # paperIdautherId = pd.read_csv('./data/dataset/PaperAuthor.csv')

    author1 = set(train['AuthorId'])
    author2 = set(valid['AuthorId'])
    print(author1.intersection(author2))

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
