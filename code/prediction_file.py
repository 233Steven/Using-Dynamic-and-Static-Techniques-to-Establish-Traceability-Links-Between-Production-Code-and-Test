import pandas as pd
import random, csv, util
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

test_tao = False

# def get_truth(pname, level="method"):
#     result = []
#     with open("result\\groundtruth_"+level+"_level_"+pname+".csv", "r") as f:
#         rows = csv.reader(f)
#         next(rows)
#         for row in rows:
#             result.append((row[0].strip(), row[1].strip()))
#     return result

def get_links(df, tech, threshold, test):
    result = []
    for row_id in range(df.shape[0]):
        if df.iloc[row_id]['test_file'].strip() in test:
            if df.iloc[row_id][tech] >= threshold:
                result.append((df.iloc[row_id][tech], df.iloc[row_id]['test_file'].strip(),df.iloc[row_id]['production_file'].strip()))
    result.sort()
    tmp = []
    for i in result:
        tmp.append((i[1], i[2]))
    return tmp

def get_links2(df, tech="NCC", threshold=1):
    result = []
    for row_id in range(df.shape[0]):
        if df.iloc[row_id][tech] >= threshold:
            r = random.randint(1, 10)
            if r >= 5:
                continue
            result.append((df.iloc[row_id]['test_file'].strip(),df.iloc[row_id]['production_file'].strip()))
    return result

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
        result.add(gtt[0])
    return result

def get_all(df, test):
    result = []
    for row_id in range(df.shape[0]):
            if df.iloc[row_id]['test_file'].strip() in test:
                result.append((df.iloc[row_id]['test_file'].strip(),df.iloc[row_id]['production_file'].strip()))
    return result

def predict_file(technique_list):
    projects_list = ["boltons", "flask", "httpie", "requests", "scrapy", "textdistance", "face_recognition"]
    predict_tao = { "NC"            : 0.10,\
                    "NCC"           : 0.30,\
                    "LCS-U"         : 0.90,\
                    "LCS-B"         : 0.60,\
                    "Leven"         : 0.60,\
                    "LCBA"          : 0.10,\
                    "Tarantula"     : 0.60,\
                    "TFIDF"         : 0.10,\
                    "Static NC"     : 0.10,\
                    "Static NCC"    : 0.10,\
                    "Static LCS-U"  : 0.90,\
                    "Static LCS-B"  : 0.90,\
                    "Static Leven"  : 0.90,\
                    "Similarity"    : 0.10,\
                    "Co-ev"         : 0.20}
    pjn = len(projects_list)
    gts = []
    tests = []
    score_alls = []
    scoredf_dict = dict()
    for pj in projects_list:
        scoredf_dict[pj] = pd.read_csv("result\\score_file_level_" + pj + ".csv")
        gt = util.get_truth(pj, "module")
        gts.extend(gt)
        test = get_test(gt)
        tests.extend(test)
        score_all = util.get_score_all_file(scoredf_dict[pj], test)
        score_alls.extend(score_all)
    print(len(gts), len(tests), len(score_alls))
    for i in range(len(technique_list)):
        tech_name = technique_list[i]
        tao = predict_tao[tech_name]
        predictions = []
        for pj in projects_list:
            prediction = get_links(scoredf_dict[pj], tech_name, tao, tests)
            predictions.extend(prediction)
        precision, recall, f1, mAP, auc, tp, fp = util.get_metric(gts, predictions, score_alls)
        print("%s & %.1f & %.1f & %.1f & %.1f & %.1f" % (tech_name, precision, recall, f1, mAP, auc))


if __name__ == "__main__":
    technique_list = util.get_technique_list()
    predict_file(technique_list)
