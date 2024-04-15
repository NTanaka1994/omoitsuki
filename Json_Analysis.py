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


url = "https://www.jma.go.jp/bosai/forecast/data/forecast/130000.json" #ここにJSONのあるURLを入れる

res = requests.get(url)

data = json.loads(res.text)

depth(data, "data")
print()
