import pandas as pd
import random, csv
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import util, xlrd

test_tao = False
 
def get_links(df, tech, threshold, test):
    result = []
    for row_id in range(df.shape[0]):
        if (df.iloc[row_id]['test_root'].strip(),df.iloc[row_id]['test_method'].strip()) in test:
            if df.iloc[row_id][tech] > threshold:
                result.append((df.iloc[row_id][tech], df.iloc[row_id]['test_root'].strip(),df.iloc[row_id]['test_method'].strip(), \
                        df.iloc[row_id]['production_root'].strip(),df.iloc[row_id]['production_method'].strip()))
    result.sort()
    tmp = []
    for i in result:
        tmp.append((i[1], i[2], i[3], i[4]))
    return tmp

def get_links2(df, tech, threshold, test):
    result = []
    for row_id in range(df.shape[0]):
        if (df.iloc[row_id]['test_root'].strip(),df.iloc[row_id]['test_method'].strip()) in test:
            if df.iloc[row_id][tech] > threshold:
                result.append((df.iloc[row_id][tech], df.iloc[row_id]['test_root'].strip(),df.iloc[row_id]['test_method'].strip(), \
                        df.iloc[row_id]['production_root'].strip(),df.iloc[row_id]['production_method'].strip()))
    result.sort()
    tmp = []
    for i in result:
        tmp.append((i[1], i[2], i[3], i[4]))
    return tmp

def write_link(links, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("test,production\n")
        for i in links:
            f.write(",".join(i)+"\n")

def cal_auc1(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def cal_map(y_true, y_pred):
    return average_precision_score(y_true, y_pred)

def get_test(gt):
    result = set()
    for gtt in gt:
        result.add((gtt[0], gtt[1]))
    return result

def get_pro(gt):
    result = set()
    for gtt in gt:
        result.add((gtt[2], gtt[3]))
    return result

def get_all(df, test):
    result = []
    for row_id in range(df.shape[0]):
            if (df.iloc[row_id]['test_root'].strip(),df.iloc[row_id]['test_method'].strip()) in test:
                result.append((df.iloc[row_id]['test_root'].strip(),df.iloc[row_id]['test_method'].strip(), \
                               df.iloc[row_id]['production_root'].strip(),df.iloc[row_id]['production_method'].strip()))
    return result
    

def prediction_rq1(pname, scoredf, technique_list):
    # ground_truth_tao = [1, 1, 0.9, 0.9, 1, 1, 0.9, 0.95, 1, 1, 1, 1, 1]
    # predict_tao = [0.5, 0.5, 0.75, 0.55, 0.95, 1, 0.95, 0.9, 1, 1, 1, 1, 0.995, 0.5, 0.5]
    predict_tao = { "NC"            : 0.30,\
                    "NCC"           : 0.10,\
                    "LCS-U"         : 0.90,\
                    "LCS-B"         : 0.80,\
                    "Leven"         : 0.80,\
                    "LCBA"          : 0.10,\
                    "Tarantula"     : 0.90,\
                    "TFIDF"         : 0.70,\
                    "Static NC"     : 0.10,\
                    "Static NCC"    : 0.10,\
                    "Static LCS-U"  : 0.90,\
                    "Static LCS-B"  : 0.90,\
                    "Static Leven"  : 0.90,\
                    "Similarity"    : 0.40,\
                    "Co-ev"         : 0.70}
    s = pname
    ss = pname
    gt = util.get_truth(pname, "method")
    test = get_test(gt)
    gt_size = len(gt)
    x = []
    score_all = get_all(scoredf, test)
    for i in score_all:
        if i in gt:
            x.append(1)
        else:
            x.append(0)
    # TEST TAO
    if test_tao:
        projects_list = ["boltons", "flask", "httpie", "requests", "scrapy", "textdistance", "face_recognition"]
        pjn = len(projects_list)
        for i in range(len(technique_list)):
            tech_name = technique_list[i]
            max_tao = 0
            max_precision = 0
            max_recall = 0
            max_f1 = 0
            max_mAP = 0
            max_auc = 0
            for pp in range(0, 101, 10):
                tao = 0.01 * pp
                precision_sum = 0
                recall_sum = 0
                f1_sum = 0
                mAP_sum = 0
                auc_sum = 0
                for pj in projects_list:
                    scoredf = pd.read_csv("result\\score_method_level_" + pj + ".csv")
                    gt = util.get_truth(pj, "method")
                    test = get_test(gt)
                    score_all = get_all(scoredf, test)
                    prediction = get_links(scoredf, tech_name, tao, test)
                    precision, recall, f1, mAP, auc, tp, fp = util.get_metric(gt, prediction, score_all)
                    precision_sum += precision
                    recall_sum += recall
                    f1_sum += f1
                    mAP_sum += mAP
                    auc_sum += auc
                if max_f1 < f1_sum / pjn:
                    max_tao = tao
                    max_precision = precision_sum / pjn
                    max_recall = recall_sum / pjn
                    max_f1 = f1_sum / pjn
                    max_mAP = mAP_sum / pjn
                    max_auc = auc_sum / pjn
            print(max_tao)
            print(",".join([tech_name, str(max_precision), str(max_recall), str(max_f1), str(max_mAP), str(max_auc)]))
    else:
        for i in range(len(technique_list)):
            tech_name = technique_list[i]
            s = s + "," + tech_name
            ss = ss + " & " + tech_name.replace("Leven", "LD")
            tao = predict_tao[tech_name]
            prediction = get_links(scoredf, tech_name, tao, test)
            precision, recall, f1, mAP, auc, tp, fp = util.get_metric(gt, prediction, score_all)
            s = s + "," + str(round(precision, 1))
            ss = ss + " & " + str(round(precision, 1))
            s = s + "," + str(round(recall, 1))
            ss = ss + " & " +  str(round(recall, 1))
            s = s + "," + str(round(f1, 1))
            ss = ss + " & " + str(round(f1, 1))
            s = s + "," + str(round(mAP, 1))
            ss = ss + " & " + str(round(mAP, 1))
            s = s + "," + str(round(auc, 1))
            if i in [0, 1, 5, 8, 9]:
                ss = ss + " & -"
            else:
                ss = ss + " & " + str(round(auc, 1))
            s = s + "," + str(tp) + "," + str(fp)  + "\n"
            ss = ss + " \\\\\n" 
        print(ss)
        with open("res.csv", "w") as f:
            f.write(s)

def predict_method(technique_list):
    projects_list = ["boltons", "flask", "httpie", "requests", "scrapy", "textdistance", "face_recognition"]
    # ground_truth_tao = [1, 1, 0.9, 0.9, 1, 1, 0.9, 0.95, 1, 1, 1, 1, 1]
    # predict_tao = [0.5, 0.5, 0.75, 0.55, 0.95, 1, 0.95, 0.9, 1, 1, 1, 1, 0.995, 0.5, 0.5]
    predict_tao = { "NC"            : 0.30,\
                    "NCC"           : 0.10,\
                    "LCS-U"         : 0.90,\
                    "LCS-B"         : 0.80,\
                    "Leven"         : 0.80,\
                    "LCBA"          : 0.10,\
                    "Tarantula"     : 0.90,\
                    "TFIDF"         : 0.70,\
                    "Static NC"     : 0.10,\
                    "Static NCC"    : 0.10,\
                    "Static LCS-U"  : 0.90,\
                    "Static LCS-B"  : 0.90,\
                    "Static Leven"  : 0.90,\
                    "Similarity"    : 0.40,\
                    "Co-ev"         : 0.70}
    pjn = len(projects_list)
    gts = []
    tests = []
    score_alls = []
    scoredf_dict = dict()
    for pj in projects_list:
        scoredf_dict[pj] = pd.read_csv("result\\score_method_level_" + pj + ".csv")
        gt = util.get_truth(pj, "method")
        gts.extend(gt)
        test = get_test(gt)
        tests.extend(test)
        score_all = util.get_score_all_method(scoredf_dict[pj], test)
        score_alls.extend(score_all)
    print(len(gts))
    for i in range(len(technique_list)):
        tech_name = technique_list[i]
        tao = predict_tao[tech_name]
        predictions = []
        for pj in projects_list:
            prediction = get_links(scoredf_dict[pj], tech_name, tao, tests)
            predictions.extend(prediction)
        precision, recall, f1, mAP, auc, tp, fp = util.get_metric(gts, predictions, score_alls)
        print(",".join([tech_name, str(precision), str(recall), str(f1), str(mAP), str(auc), str(tp), str(fp)]))
    
if __name__ == "__main__":
    pname = "face_recognition"
    scoredf = pd.read_csv("result\\score_method_level_" + pname + ".csv")
    technique_list = util.get_technique_list()
    # prediction_rq1(pname, scoredf, technique_list)
    predict_method(technique_list)
