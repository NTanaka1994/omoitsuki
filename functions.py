from sklearn.tree import export_graphviz as EG
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
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
