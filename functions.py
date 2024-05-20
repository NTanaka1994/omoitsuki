from sklearn.tree import export_graphviz as EG
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import classification_report
from pydotplus import graph_from_dot_data as GFDD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

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

def effecttest2(df, columns, y_name, auto=False):
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
    if auto:
        for i in range(len(dtr)):
            for j in range(len(dtr[i])):
                for k in range(j, len(dtr[i])):
                    if j != k:
                        f, p = stats.bartlett(dtr[i][0], dtr[i][1])
                        if (2 * p) <= 0.05:
                            t, p = stats.ttest_ind(dtr[i][j], dtr[i][k], equal_var=False)
                        else:
                            t, p = stats.ttest_ind(dtr[i][j], dtr[i][k], equal_var=True)
                        print(lab[i][j], lab[i][k])
                        print("t   = %f"%(t))
                        print("p   = %f"%(p))
                        print("val = %f"%(ave[i][j]-ave[i][k]))
                        print()
        for i in range(len(dtr)):
            plt.boxplot(dtr[i], positions=pos[i], labels=lab[i])
            plt.plot(pos[i], ave[i], marker="x")
        plt.xticks(rotation=90)
        plt.show()
    return ave, dtr, pos , lab

def corr_sort(df):
    corr = df.corr().values
    dst = []
    for i in range(0, len(df.corr().columns)):
        for j in range(0, i):
            if i != j:
                dst.append([df.corr().columns[i], df.corr().columns[j], corr[i][j]])
    df_corr = pd.DataFrame(dst)
    df_corr.columns = ["因子1", "因子2", "相関係数"]
    return df_corr.sort_values("相関係数", ascending=False)

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

def adjr2_score(y_test, y_pred, x):
    return 1 - (1 - r2_score(y_test, y_pred)) * (len(y_test) - 1) / (len(y_test) - len(x.columns) - 1)

def regression_report(y_test, y_pred, x):
    print("R2      : %f"%(r2_score(y_test, y_pred)))
    print("adjR2   : %f"%(adjr2_score(y_test, y_pred, x)))
    print("MSE     : %f"%(mean_squared_error(y_test, y_pred)))
    print("RMSE    : %f"%(np.sqrt(mean_squared_error(y_test, y_pred))))
    try:
        print("RMSLE   : %f"%(np.sqrt(mean_squared_log_error(y_test, y_pred, squared=False))))
    except:
        _ = 0
    print("MAE     : %f"%(mean_absolute_error(y_test, y_pred)))
    pass

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

def showline_clf(x, y, model, modelname, x0="x0", x1="x1"):
    fig, ax = plt.subplots(figsize=(8, 6))
    X, Y = np.meshgrid(np.linspace(*ax.get_xlim(), 1000), np.linspace(*ax.get_ylim(), 1000))
    XY = np.column_stack([X.ravel(), Y.ravel()])
    x = preprocessing.minmax_scale(x)
    model.fit(x, y)
    Z = model.predict(XY).reshape(X.shape)
    plt.contourf(X, Y, Z, alpha=0.1, cmap="brg")
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap="brg")
    plt.xlim(min(x[:, 0]), max(x[:, 0]))
    plt.ylim(min(x[:, 1]), max(x[:, 1]))
    plt.title(modelname)
    plt.colorbar()
    plt.xlabel(x0)
    plt.ylabel(x1)
    plt.show()

def ci_plot(df, per=0.95):
    x = 0
    for col in df.columns:
        var = df[col].var()
        mean = df[col].mean()
        trange = stats.t(loc=mean, scale=np.sqrt(var/len(df[col].values)), df=len(df[col].values)-1)
        low, high = trange.interval(per)
        print(col)
        print("%f <= z <= %f"%(low, high))
        plt.plot([low, high], [x, x], marker="|", color="#000000")
        plt.scatter(mean, x, marker="x", color="#000000")
        x = x + 1
    plt.yticks(np.arange(len(df.columns)), df.columns)
    plt.show()

def boxplot(df):
    x = 0
    for i in range(len(df.columns)):
        plt.boxplot(df[df.columns[i]], positions=[x], labels=[df.columns[i]])
        plt.scatter(x, df[df.columns[i]].mean(), marker="x", color="#000000")
        x = x + 1
    plt.xticks(rotation=90)
    plt.show()

def marcov(arr):
    dic = {}
    for i in range(1, len(arr)):
        if arr[i-1] in dic:
            if arr[i] in dic[arr[i-1]]:
                dic[arr[i-1]][arr[i]] = dic[arr[i-1]][arr[i]] + 1
            else:
                dic[arr[i-1]][arr[i]] = 1
        else:
            dic[arr[i-1]] = {}
            dic[arr[i-1]][arr[i]] = 1
    col = []
    sumval = {}
    for val in dic:
        col.append(val)
        for val2 in dic[val]:
            col.append(val2)
            try:
                sumval[val] = sumval[val] + dic[val][val2]
            except:
                sumval[val] = dic[val][val2]
    col = list(set(col))
    for val in dic:
        for val2 in dic[val]:
            dic[val][val2] = dic[val][val2] / sumval[val]
    table = []
    for val in col:
        tmp = []
        for val2 in col:
            try:
                tmp.append(dic[val][val2])
            except:
                tmp.append(0)
        table.append(tmp)
    df = pd.DataFrame(table)
    df.columns = col
    df.index = col
    return df, dic

def marcov_pred(dic, start, length):
    state = start
    flow = [state]
    for i in range(length):
        if state in dic:
            next_state = list(dic[state].keys())
            proba = list(dic[state].values())
            state = np.random.choice(next_state, p=proba)
        else:
            break
        flow.append(state)
    return flow

def cart_analysis_GBDT(x, y, model, filename="CART"):
    y_data = list(set(y.values))
    for i in range(len(y_data)):
        y_data[i] = str(y_data[i])
    for i in range(len(model.estimators_[len(model.estimators_)-1])):
        dotdata = EG(model.estimators_[len(model.estimators_)-1, i], filled=True, rounded=True, class_names=y_data, feature_names=x.columns, out_file=None)
        graph = GFDD(dotdata)
        graph.write_png(filename+str(i)+".png")

def cart_analysis_RF(x, y, model, filename="CART"):
    y_data = list(set(y.values))
    for i in range(len(y_data)):
        y_data[i] = str(y_data[i])
    for i in range(len(model.estimators_)):
        dotdata = EG(model.estimators_[i], filled=True, rounded=True, class_names=y_data, feature_names=x.columns, out_file=None)
        graph = GFDD(dotdata)
        graph.write_png(filename+str(i)+".png")

def cart_analysis_DT(x, y, model, filename="CART"):
    y_data = list(set(y.values))
    for i in range(len(y_data)):
        y_data[i] = str(y_data[i])
    dotdata = EG(model, filled=True, rounded=True, class_names=y_data, feature_names=x.columns, out_file=None)
    graph = GFDD(dotdata)
    graph.write_png(filename+".png")

def classification_report_dataframe(y, y_pred):
    rep = classification_report(y, y_pred, output_dict=True)
    dst = []
    index = []
    for col in rep:
        index.append(col)
        if col != "accuracy":
            dst.append([rep[col]["precision"], rep[col]["recall"], rep[col]["f1-score"],rep[col]["support"]])
        else:
            dst.append(["", "", rep["accuracy"], rep["weighted avg"]["support"]])
    df_rep = pd.DataFrame(dst)
    df_rep.index = index
    df_rep.columns = ["precision", "recall", "f1-score", "support"]
    return df_rep

def optimal_threshold(y, pred, n="0.0", p="1.0"):
    tsd = np.linspace(0.1, 0.9, 100)
    accs = []
    rec0 = []
    rec1 = []
    pcs0 = []
    pcs1 = []
    f1_0 = []
    f1_1 = []
    for i in range(len(tsd)):
        y_pred = np.where(pred >= tsd[i], 1, 0)
        rep = classification_report(y, y_pred, output_dict=True)
        accs.append(rep["accuracy"])
        rec0.append(rep[n]["recall"])
        rec1.append(rep[p]["recall"])
        pcs0.append(rep[n]["precision"])
        pcs1.append(rep[p]["precision"])
        f1_0.append(rep[n]["f1-score"])
        f1_1.append(rep[p]["f1-score"])
    plt.plot(tsd, accs, label="accuracy")
    plt.plot(tsd, rec0, label="recall 0")
    plt.plot(tsd, rec1, label="recall 1")
    plt.plot(tsd, pcs0, label="precision 0")
    plt.plot(tsd, pcs1, label="precision 1")
    plt.plot(tsd, f1_0, label="f1-score 0")
    plt.plot(tsd, f1_1, label="f1-score 1")
    plt.legend()
    plt.show()
    return tsd[np.argmax(accs)], tsd[np.argmax(rec0)], tsd[np.argmax(rec1)], tsd[np.argmax(pcs0)], tsd[np.argmax(pcs1)], tsd[np.argmax(f1_0)], tsd[np.argmax(f1_1)]
