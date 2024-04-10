from collections import deque
import json
import requests

def width(jsn, var=""):
    queue = deque([(jsn, var)])
    while queue:
        node, var = queue.popleft()
        if isinstance(node, dict):
            for key in node:
                queue.append((node[key], var + "[\""+key+"\"]"))
        elif isinstance(node, list) and node != []:
            for i in range(len(node)):
                queue.append((node[i], var + "["+str(i)+"]"))
        else:
            print(var + "=" + str(node))
            pass


url = "https://raw.githubusercontent.com/NTanaka1994/DataAnalysis/main/JSON%E3%81%AE%E8%A7%A3%E6%9E%90/json_sample.json" #ここにJSONのあるURLを入れる

res = requests.get(url)

data = json.loads(res.text)

width(data, "data")
print()
