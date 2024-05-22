import pandas as pd
import random, csv, util
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

test_tao = False

def get_truth(pname, level="method"):
    result = []
    with open("result\\groundtruth_"+level+"_level_"+pname+".csv", "r") as f:
        rows = csv.reader(f)
        next(rows)
        for row in rows:
            result.append((row[0].strip(), row[1].strip()))
    return result

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
                    "LCS-B"         : 0.70,\
                    "Leven"         : 0.80,\
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
        print(",".join([tech_name, str(precision), str(recall), str(f1), str(mAP), str(auc), str(tp), str(fp)]))

if __name__ == "__main__": 
    pname = "face_recognition"
    scoredf = pd.read_csv("score\\score_file_level_" + pname + ".csv")
    technique_list = util.get_technique_list()
    predict_file(technique_list)
    # ground_truth_tao = [1, 1, 0.9, 0.9, 1, 1, 0.9, 0.95, 1, 1, 1, 1, 1]
    # predict_tao = { "NC"            : 0.10,\
    #                 "NCC"           : 0.30,\
    #                 "LCS-U"         : 0.90,\
    #                 "LCS-B"         : 0.70,\
    #                 "Leven"         : 0.80,\
    #                 "LCBA"          : 0.10,\
    #                 "Tarantula"     : 0.60,\
    #                 "TFIDF"         : 0.10,\
    #                 "Static NC"     : 0.10,\
    #                 "Static NCC"    : 0.10,\
    #                 "Static LCS-U"  : 0.90,\
    #                 "Static LCS-B"  : 0.90,\
    #                 "Static Leven"  : 0.90,\
    #                 "Similarity"    : 0.10,\
    #                 "Co-ev"         : 0.20}
    # gt = util.get_truth(pname, "module")
    # test = get_test(gt)
    # gt_size = len(gt)
    # x = []
    # score_all = get_all(scoredf, test)
    # for i in score_all:
    #     if i in gt:
    #         x.append(1)
    #     else:
    #         x.append(0)
    # s = pname
    # ss = pname
    # projects_list = ["boltons", "flask", "httpie", "requests", "scrapy", "textdistance", "face_recognition"]
    # if test_tao:
    #     pjn = len(projects_list)
    #     for i in range(len(technique_list)):
    #         tech_name = technique_list[i]
    #         max_tao = 0
    #         max_precision = 0
    #         max_recall = 0
    #         max_f1 = 0
    #         max_mAP = 0
    #         max_auc = 0
    #         for pp in range(0, 101, 10):
    #             tao = 0.01 * pp
    #             precision_sum = 0
    #             recall_sum = 0
    #             f1_sum = 0
    #             mAP_sum = 0
    #             auc_sum = 0
    #             for pj in projects_list:
    #                 scoredf = pd.read_csv("result\\score_file_level_" + pj + ".csv")
    #                 gt = util.get_truth(pj, "module")
    #                 test = get_test(gt)
    #                 score_all = get_all(scoredf, test)
    #                 prediction = get_links(scoredf, tech_name, tao, test)
    #                 precision, recall, f1, mAP, auc, tp, fp = util.get_metric(gt, prediction, score_all)
    #                 precision_sum += precision
    #                 recall_sum += recall
    #                 f1_sum += f1
    #                 mAP_sum += mAP
    #                 auc_sum += auc
    #             if max_f1 < f1_sum / pjn:
    #                 max_tao = tao
    #                 max_precision = precision_sum / pjn
    #                 max_recall = recall_sum / pjn
    #                 max_f1 = f1_sum / pjn
    #                 max_mAP = mAP_sum / pjn
    #                 max_auc = auc_sum / pjn
    #         print(max_tao)
    #         print(",".join([tech_name, str(max_precision), str(max_recall), str(max_f1), str(max_mAP), str(max_auc)]))
    # else:
    #     for i in range(len(technique_list)):
    #         tech_name = technique_list[i]
    #         tao = predict_tao[tech_name]
    #         s = s + "," + technique_list[i]
    #         ss = ss + " & " + technique_list[i].replace("Leven", "LD")
    #         prediction = get_links(scoredf, tech_name, tao, test)
    #         prediction_size = len(prediction)
    #         y = []
    #         for j in score_all:
    #             if j in prediction:
    #                 y.append(1)
    #             else:
    #                 y.append(0)
    #         tp = len(set(prediction) & set(gt))
    #         fp = prediction_size - tp
    #         fn = gt_size - tp
    #         try:
    #             precision = tp/(tp+fp)
    #         except:
    #             precision = 0
    #         s = s + "," + str(round(precision*100, 1))
    #         ss = ss + " & " + str(round(precision*100, 1))
    #         recall = tp/(tp+fn)
    #         s = s + "," + str(round(recall*100, 1))
    #         ss = ss + " & " +  str(round(recall*100, 1))
    #         try:
    #             f1 = 2*precision*recall/(precision+recall)
    #         except:
    #             f1 = 0
    #         s = s + "," + str(round(f1*100, 1))
    #         ss = ss + " & " + str(round(f1*100, 1))
    #         mAP = cal_map(x, y)
    #         s = s + "," + str(round(mAP*100, 1))
    #         ss = ss + " & " + str(round(mAP*100, 1))
    #         try:
    #             auc = cal_auc1(x, y)
    #         except:
    #             auc = 0
    #         s = s + "," + str(round(auc*100, 1))
    #         if i in [0, 1, 5, 8, 9]:
    #             ss = ss + " & -"
    #         else:
    #             ss = ss + " & " + str(round(auc*100, 1))
    #         s = s + "," + str(tp) + "," + str(fp)  + "\n"
    #         ss = ss + " \\\\\n" 
    #     print(ss)
    #     with open("res.csv", "w") as f:
    #         f.write(s)