import pandas as pd
dataDf = pd.read_csv("data.csv")
# typeUnq = dataDf["type"].unique()
trainTxt = "data/train.txt"
testTxt = "data/test.txt"
vailTxt = "data/valid.txt"
diseaseCount = len(dataDf)
shuffled_df = dataDf.sample(frac=1).reset_index(drop=True)
trainDf = shuffled_df[: int(0.7*diseaseCount)]
vaildDf = shuffled_df[int(0.7*diseaseCount):int(0.9*diseaseCount)]
testDf = shuffled_df[int(0.9*diseaseCount):]
# 将实体和关系进行编号
dataList = dataDf["lEntity"].to_list() + dataDf["rEntity"].to_list()
dataList = list(set(dataList))
dataSerise = pd.Series(dataList)
dataSerise.to_csv("data/entities.tsv",sep="\t", header=None)
relaList = dataDf["Rela"].unique().tolist()
relaSerise = pd.Series(relaList)
relaSerise.to_csv("data/relations.tsv",sep="\t", header=None)
trainDf[['lEntity','Rela','rEntity']].to_csv(trainTxt, sep="\t", index=False, header=None)
testDf[['lEntity','Rela','rEntity']].to_csv(testTxt, sep="\t", index=False, header=None)
vaildDf[['lEntity','Rela','rEntity']].to_csv(vailTxt, sep="\t", index=False, header=None)

# for typeDiseas in typeUnq:
#     diseaseDf = dataDf[dataDf['type']==typeDiseas].copy()
#     shuffled_df = diseaseDf.sample(frac=1).reset_index(drop=True)
#     diseaseCount = len(shuffled_df)
#     trainDf = shuffled_df[: int(0.8*diseaseCount)]
#     testDf = shuffled_df[int(0.8*diseaseCount):]
#     trainDf.to_csv(trainTxt, sep="\t", index=False)
#     testDf.to_csv(testTxt, sep="\t", index=False)
#     break