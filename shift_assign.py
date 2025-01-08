import pulp
import pandas as pd

df_ava = pd.read_csv("shift_ava.csv", sep="\t")
df_req = pd.read_csv("shift_req.csv")
df_ava2 = df_ava.copy()
df_ava2.index = df_ava["Time"]
df_ava2 = df_ava2.drop("Time", axis=1)
df_off = pd.read_csv("shift_off.csv")

holiday = [7, 14, 21, 28]
days = list(range(1, 32))
incompatible_pair = [("A", "B")]

emp = df_ava2.columns
shift = df_ava["Time"].values
ava = {}
for col in df_ava2.columns:
    ind = df_ava2.index
    ava[col] = {ind[0]: df_ava2.loc[ind[0], col], ind[1] : df_ava2.loc[ind[1], col], ind[2] : df_ava2.loc[ind[2], col]}
req = dict(df_req.values)
off = {}
for i in range(len(df_off.values)):
    try:
        off[df_off.values[i][0]].append(df_off.values[i][1])
    except:
        off[df_off.values[i][0]] = [df_off.values[i][1]]

prob = pulp.LpProblem("shift", pulp.LpMinimize)

# 変数の定義
x = pulp.LpVariable.dicts("shift_num", [(e, s, d) for e in emp for s in shift for d in days], cat="Binary")

# 目的関数: 希望に沿わないシフトの割り当てを最小化
prob += pulp.lpSum((1-ava[e][s]*x[(e, s, d)] for e in emp for s in shift for d in days))

# 制約1　各シフトに必要な最低人数を確保
for d in days:
    if d in holiday:
        for s in shift:
            prob += pulp.lpSum(x[e, s, d] for e in emp) == 0
    else:
        for s in shift:
            prob += pulp.lpSum(x[e, s, d] for e in emp) >= req[s]

# 制約2　各従業員は1日に2つまでのシフトにしか入れない
for e in emp:
    for d in days:
        prob += pulp.lpSum(x[e, s, d] for s in shift) <= 2

# 制約3　希望休を考慮
for e, days_off in off.items():
    for d in days_off:
        prob += pulp.lpSum(x[e, s, d] for s in shift) == 0

# 制約4　ペアを組みたくない人を考慮
for d in days:
    for s in shift:
        for pair in incompatible_pair:
            e1, e2 = pair
            prob += x[(e1, s, d)] + x[(e2, s, d)] <= 1

# 解を求める
prob.solve()
print("Status:", pulp.LpStatus[prob.status])
print("Assign:")
for d in days:
    print("Day %d"%(d))
    for s in shift:
        assign = []
        for e in emp:
            if x[(e, s, d)].value() == 1:
                assign.append(e)
        print(s, assign) if assign else "No one assign"
