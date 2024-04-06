import json
import requests

def depth(jsn, var=""):
    if isinstance(jsn, dict):
        for row in jsn:
            depth(jsn[row], var=var+"[\""+row+"\"]")
    elif isinstance(jsn, list) and jsn != []:
        for i in range(len(jsn)):
            depth(jsn[i], var=var+"["+str(i)+"]")
    else:
        print(var+"="+str(jsn))
        pass


url = "https://raw.githubusercontent.com/NTanaka1994/DataAnalysis/main/JSON%E3%81%AE%E8%A7%A3%E6%9E%90/json_sample.json" #ここにJSONのあるURLを入れる

res = requests.get(url)

data = json.loads(res.text)

depth(data, "data")
print()
