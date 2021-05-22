from sklearn.datasets import load_svmlight_file
from sklearn import svm
import pandas as pd


if __name__ == '__main__':
    paper = pd.read_csv('./data/dataset/Paper.csv')
    paperIdautherId = pd.read_csv('./data/dataset/PaperAuthor.csv')

    paperId = 1291787
    year = paper[paper['Id'] == paperId]['Year'].values[0]
    print(year)
    # 根据 authorId 找到发表过的所有文章的 paperId
    # authorId = 1140870
    # paperIds = paperIdautherId[paperIdautherId['AuthorId'] == int(authorId)]['PaperId'].values
    # paperYears = []
    # for id in paperIds:
    #     year = paper[paper['Id'] == id]['Year'].values[0]
    #     if 1800 <= year <= 2013:
    #         paperYears.append(year)

    # print(sum(paperYears) / len(paperYears))