from sklearn.tree import export_graphviz as EG
from pydotplus import graph_from_dot_data as GFDD
import pandas as pd
import numpy as np

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

def effecttest(df, columns, y_name):
    ave = []
    dtr = []
    pos = []
    lab = []
    x = 0
    for col in columns:
        values = list(set(df[col].values))
        tmp_ave = []
        tmp_dtr = []
        tmp_pos = []
        tmp_lab = []
        for val in values:
            df_tmp = df[df[col]==val]
            tmp_ave.append(df_tmp[y_name].mean())
            tmp_dtr.append(df_tmp[y_name])
            tmp_pos.append(x)
            tmp_lab.append(col+"_"+str(val))
            x = x + 1
        ave.append(tmp_ave)
        dtr.append(tmp_dtr)
        pos.append(tmp_pos)
        lab.append(tmp_lab)
    return ave, dtr, pos, lab

def pcorr(df):
    pcor = []
    corr = df.corr()
    columns = df.corr().columns
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            for k in range(len(columns)):
                if columns[i] != columns[k] and columns[j] != columns[k]:
                    xy = corr.loc[columns[i], columns[j]]
                    yz = corr.loc[columns[j], columns[k]] 
                    zx = corr.loc[columns[k], columns[i]]
                    pc = (xy - yz * zx) / (np.sqrt(1-yz**2) * np.sqrt(1-zx**2))
                    pcor.append([columns[i], 
                                columns[j], 
                                columns[k], 
                                corr.loc[columns[i], columns[j]],
                                pc,
                                abs(corr.loc[columns[i], columns[j]]-pc)])
    df_pcor = pd.DataFrame(pcor)
    df_pcor.columns = ["因子1", "因子2", "第三の因子", "相関係数", "偏相関係数", "偏相関係数と相関係数の差"]
    return df_pcor

def miss_pred(y_test, y_pred, model):
    miss = []
    y_predSM = model.predict_proba(x_test)
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            miss.append([i, y_test[i], y_pred[i], max(y_predSM[i])])
    df_miss = pd.DataFrame(miss)
    df_miss.columns = ["ID", "real", "predict", "probably"]
    return df_miss

def factor_exp(pca, x):
    col = []
    for i in range(len(x.columns)):
        col.append("第%d主成分"%(i+1))
    exp = pca.explained_variance_ratio_
    df_exp = pd.DataFrame(exp)
    df_exp.index = col
    df_exp.columns = ["寄与率"]
    df_exp_sum = df_exp.cumsum()
    com = pca.components_
    fac = []
    for i in range(len(exp)):
        fac.append(np.sqrt(exp[i])*com[i])
    df_fac = pd.DataFrame(fac)
    df_fac.columns = x.columns
    df_fac.index = col
    return df_fac, df_exp_sum

def x_importance(x, y, dtc):
    imp = dtc.feature_importances_
    df_imp = pd.DataFrame(imp)
    df_imp.index = x.columns
    df_imp.columns = ["var_imp"]
    y_data = list(set(y.values))
    for i in range(len(y_data)):
        y_data[i] = str(y_data[i])
    dotdata = EG(dtc,filled=True, rounded=True, class_names=y_data, feature_names=x.columns, out_file=None)
    graph = GFDD(dotdata)
    graph.write_png("CART.png")
    return df_imp
