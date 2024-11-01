import os
import pandas as pd

if __name__ == '__main__':
    filePath = 'KLGraph/results'
    listDir = os.listdir(filePath)
    result = []
    for dir in listDir:
        with open(os.path.join(filePath, dir), "r") as f:
            lines = f.readlines()
            for line in lines:
                data = line.split(". (")[-1].replace(")","").replace("\n","").split(",")
                result.append([data[0].strip(), data[1].strip(), data[2].strip()])
        pd.DataFrame(result, columns=["lEntity", "Rela", "rEntity"]).to_csv(os.path.join("output/rela",dir.replace(".txt", ".csv")), index=False)

