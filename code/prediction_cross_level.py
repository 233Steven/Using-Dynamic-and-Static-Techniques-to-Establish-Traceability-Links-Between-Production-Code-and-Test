from collections import defaultdict
import pandas as pd
import random, csv, util
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

test_tao = False

def get_score_method(test_files, function_in_test, df, technique_list):
    score = dict()
    for tech in technique_list:
        score[tech] = defaultdict()
        for test_file in test_files:
            for func in function_in_test[test_file]:
                test_module = test_file + '@' + func
                score[tech][test_module] = defaultdict(float)
    # n = df.shape[0]
    # cnt = 0
    for row in df.itertuples():
        # if cnt % 10000 == 0:
        #     print(cnt, n)
        # cnt+=1
        test_module = str(row.test_root) + "@" + str(row.test_method)
        production_module = str(row.production_root) + "@" + str(row.production_method)
        if not row.test_method.startswith("test_"):
            continue
        for i in range(len(technique_list)):
            tech = technique_list[i]
            try: 
                score[tech][test_module][production_module] = float(row[5+i])
            except:
                pass
    return score

def get_score_file(test_files, production_files, df, technique_list):
    score = dict()
    for tech in technique_list:
        score[tech] = defaultdict()
        for test_file in test_files:
            score[tech][test_file] = defaultdict(float)
    for row in df.itertuples():
        test_file = str(row.test_file)
        production_file = str(row.production_file) 
        for i in range(len(technique_list)):
            tech = technique_list[i]
            try: 
                score[tech][test_file][production_file] = float(row[3+i])
            except:
                pass
    # for row_id in range(df.shape[0]):
    #     for tech in technique_list:
    #         try:
    #             score[tech][df.iloc[row_id]['test_file'].strip()][df.iloc[row_id]['production_file'].strip()] = float(df.iloc[row_id][tech])
    #         except:
    #             pass
    return score

def get_test1(gt):
    result = set()
    for gtt in gt:
        result.add(gtt[0])
    return result

def get_test2(gt):
    result = set()
    for gtt in gt:
        result.add((gtt[0], gtt[1]))
    return result

def prediction_method_to_file(technique_list):
    # predict_tao = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    projects_list = ["boltons", "flask", "httpie", "requests", "scrapy", "textdistance", "face_recognition"]
    predict_tao = { "NC"            : 0.90,\
                    "NCC"           : 0.90,\
                    "LCS-U"         : 0.90,\
                    "LCS-B"         : 0.90,\
                    "Leven"         : 0.90,\
                    "LCBA"          : 0.70,\
                    "Tarantula"     : 0.90,\
                    "TFIDF"         : 0.90,\
                    "Static NC"     : 0.10,\
                    "Static NCC"    : 0.10,\
                    "Static LCS-U"  : 0.80,\
                    "Static LCS-B"  : 0.90,\
                    "Static Leven"  : 0.90,\
                    "Similarity"    : 0.90,\
                    "Co-ev"         : 0.90}
    gts = []
    tests = []
    score_alls = []
    scoredf_file_list = []
    scoredf_method_list = []
    test_files = []
    production_files = []
    for pj in projects_list:
        project_root = "F:\\myPythonProjects\\projects_new\\" + pj
        scoredf_file_list.append(pd.read_csv("result\\score_file_level_" + pj + ".csv"))
        scoredf_method_list.append(pd.read_csv("result\\score_method_level_" + pj + ".csv"))
        gt = util.get_truth(pj, "module")
        gts.extend(gt)
        test = get_test1(gt)
        tests.extend(test)
        tf = util.get_files_from_gt(gt, 0)
        test_files.extend(tf)
        pf = util.get_files_from_gt(gt, 1)
        production_files.extend(pf)
    
    scoredf_file = pd.concat(scoredf_file_list)
    scoredf_method = pd.concat(scoredf_method_list)
    score_alls = util.get_score_all_file(scoredf_file, tests)

    print(len(gts), len(tests), len(score_alls))
    function_in_test, test_function_sum = util.get_function_in_files(test_files, True)
    function_in_production, production_function_sum = util.get_function_in_files(production_files, False)
    method_score = get_score_method(test_files, function_in_test, scoredf_method,technique_list)
    
    file_score = dict()
    cnt = dict()
    for tech in technique_list:
        file_score[tech] = dict()
        cnt[tech] = dict()
        for test_file in test_files:
            file_score[tech][test_file] = defaultdict(float)
            cnt[tech][test_file] = defaultdict(float)
            for production_file in production_files:
                file_score[tech][test_file][production_file] = 0.0
                cnt[tech][test_file][production_file] = 0
    max_score = defaultdict(float)
    for tech in technique_list:
        max_score[tech] = 0.0
        for test_file in test_files:
            for tfunc in function_in_test[test_file]:
                test_module = test_file + "@" + tfunc
                for production_file in production_files:
                    for pfunc in function_in_production[production_file]:
                        production_module = production_file + "@" + pfunc
                        file_score[tech][test_file][production_file] += method_score[tech][test_module][production_module]
                        cnt[tech][test_file][production_file] += 1
    for tech in technique_list:
        for test_file in test_files:
            mx_score = 0.0
            for production_file in production_files:
                max_score[tech] = max(max_score[tech], file_score[tech][test_file][production_file])
                mx_score = max(mx_score, file_score[tech][test_file][production_file])
            for production_file in production_files:
                try:
                    file_score[tech][test_file][production_file] /= mx_score
                except:
                    file_score[tech][test_file][production_file] = 0
    for tech in technique_list:
        print(tech, max_score[tech])
    for i in range(len(technique_list)):
        tech_name = technique_list[i]
        max_tao = 0
        max_precision = 0
        max_recall = 0
        max_f1 = 0
        max_mAP = 0
        max_auc = 0
        # for pp in range(0, 101, 10):
        #     tao = 0.01 * pp
        tao = predict_tao[tech_name]
        predictions = []
        for test_file in test_files:
            for production_file in production_files:
                if file_score[tech_name][test_file][production_file] > tao:
                    predictions.append((test_file, production_file))
        precision, recall, f1, mAP, auc, tp, fp = util.get_metric(gts, predictions, score_alls)
        #     if max_f1 < f1:
        #         max_tao = tao
        #         max_precision = precision
        #         max_recall = recall
        #         max_f1 = f1
        #         max_mAP = mAP
        #         max_auc = auc
        # print(",".join([tech_name, str(max_precision), str(max_recall), str(max_f1), str(max_mAP), str(max_auc), str(max_tao)]))
        print(",".join([tech_name, str(precision), str(recall), str(f1), str(mAP), str(auc), str(tp), str(fp)]))
    # TEST TAO
    # if test_tao:
    #     for i in range(len(technique_list)):
    #         tech_name = technique_list[i]
    #         max_pre = 0
    #         max_tao = 0
    #         max_score = ""
    #         for pp in range(0, 101, 5):
    #             tao = 0.01 * pp
    #             ss = tech_name.replace("Leven", "LD")
    #             prediction = []
    #             for test_file in test_files:
    #                 for production_file in production_files:
    #                     if file_score[tech_name][test_file][production_file] >= tao:
    #                         prediction.append((test_file, production_file))
    #             prediction_size = len(prediction)
    #             y = []
    #             for j in score_all:
    #                 if j in prediction:
    #                     y.append(1)
    #                 else:
    #                     y.append(0)
    #             tp = len(set(prediction) & set(gt))
    #             fp = prediction_size - tp
    #             fn = gt_size - tp
    #             try:
    #                 precision = tp/(tp+fp)
    #             except:
    #                 precision = 0
    #             ss = ss + " & " + str(round(precision*100, 1))
    #             recall = tp/(tp+fn)
    #             ss = ss + " & " +  str(round(recall*100, 1))
    #             try:
    #                 f1 = 2*precision*recall/(precision+recall)
    #             except:
    #                 f1 = 0
    #             ss = ss + " & " + str(round(f1*100, 1))
    #             mAP = util.cal_map(x, y)
    #             ss = ss + " & " + str(round(mAP*100, 1))
    #             try:
    #                 auc = util.cal_auc1(x, y)
    #             except:
    #                 auc = 0
    #             if i in [0, 1, 5, 8, 9]:
    #                 ss = ss + " & -"
    #             else:
    #                 ss = ss + " & " + str(round(auc*100, 1))
    #             ss = ss + "| " + str(tp) + "," + str(fp) + " \\\\\n" 
    #             if max_pre < round(precision*100, 1):
    #                 max_pre = round(precision*100, 1)
    #                 max_tao = tao
    #                 max_score = ss
    #         print(max_tao)
    #         print(max_score)
    # # NORMAL RUN
    # else:
    #     for i in range(len(technique_list)):
    #         tech = technique_list[i]
    #         tao = predict_tao[tech]
    #         s = s + "," + technique_list[i]
    #         ss = ss + " & " + technique_list[i].replace("Leven", "LD")
    #         prediction = []
    #         for test_file in test_files:
    #             for production_file in production_files:
    #                 if file_score[tech][test_file][production_file] >= tao:
    #                     prediction.append((test_file, production_file))
    #         prediction = list(set(prediction))
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
    #         mAP = util.cal_map(x, y)
    #         s = s + "," + str(round(mAP*100, 1))
    #         ss = ss + " & " + str(round(mAP*100, 1))
    #         try:
    #             auc = util.cal_auc1(x, y)
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

def prediction_file_to_method(technique_list):
    # predict_tao = [1, 1, 0.75, 0.55, 0.95, 1, 0.95, 0.9, 1, 1, 1, 1, 0.995]
    projects_list = ["boltons", "flask", "httpie", "requests", "scrapy", "textdistance", "face_recognition"]
    predict_tao = { "NC"            : 0.10,\
                    "NCC"           : 0.10,\
                    "LCS-U"         : 0.90,\
                    "LCS-B"         : 0.90,\
                    "Leven"         : 0.90,\
                    "LCBA"          : 0.10,\
                    "Tarantula"     : 0.90,\
                    "TFIDF"         : 0.10,\
                    "Static NC"     : 0.10,\
                    "Static NCC"    : 0.10,\
                    "Static LCS-U"  : 0.90,\
                    "Static LCS-B"  : 0.90,\
                    "Static Leven"  : 0.90,\
                    "Similarity"    : 0.90,\
                    "Co-ev"         : 0.90}
    gts = []
    tests = []
    score_alls = []
    scoredf_file_list = []
    scoredf_method_list = []
    production_files = []
    for pj in projects_list:
        proot = "F:\\myPythonProjects\\projects_new\\" + pj
        scoredf_file_list.append(pd.read_csv("result\\score_file_level_" + pj + ".csv"))
        scoredf_method_list.append(pd.read_csv("result\\score_method_level_" + pj + ".csv"))
        gt = util.get_truth(pj, "method")
        gts.extend(gt)
        test = get_test2(gt)
        tests.extend(test)
        pf = util.get_production_files(proot)
        production_files.extend(pf)
    scoredf_file = pd.concat(scoredf_file_list)
    scoredf_method = pd.concat(scoredf_method_list)
    score_alls = util.get_score_all_method(scoredf_method, tests)
    test_files, function_in_test, _, _ = util.get_file_and_method_from_gt(gts)
    # function_in_test, test_function_sum = util.get_function_in_files(test_files, True)
    function_in_production, production_function_sum = util.get_function_in_files(production_files, False)
    method_score = get_score_method(test_files, function_in_test, scoredf_method,technique_list)
    file_score = get_score_file(test_files, production_files, scoredf_file,technique_list)
    max_score = defaultdict(float)
    for tech in technique_list:
        max_score[tech] = 0.0
        for test_file in test_files:
            for tfunc in function_in_test[test_file]:
                test_module = test_file + "@" + tfunc
                mx_score = 0.0
                for production_file in production_files:
                    for pfunc in function_in_production[production_file]:
                        production_module = production_file + "@" + pfunc
                        # if tech == "LCS-U":
                        #     print(tech, test_module, production_module, method_score[tech][test_module][production_module], file_score[tech][test_file][production_file])
                        method_score[tech][test_module][production_module] *= file_score[tech][test_file][production_file]
                        max_score[tech] = max(max_score[tech], method_score[tech][test_module][production_module])
                        mx_score = max(mx_score, method_score[tech][test_module][production_module])
                
                for production_file in production_files:
                    for pfunc in function_in_production[production_file]:
                        production_module = production_file + "@" + pfunc
                        try:
                            method_score[tech][test_module][production_module] /= mx_score
                        except:
                            method_score[tech][test_module][production_module] = 0
    # for tech in technique_list:
    #     print(tech, max_score[tech])
    for i in range(len(technique_list)):
        tech_name = technique_list[i]
        max_tao = 0
        max_precision = 0
        max_recall = 0
        max_f1 = 0
        max_mAP = 0
        max_auc = 0
        max_tp = 0
        max_fp = 0
        for pp in range(20, 91, 10):
            tao = 0.01 * pp
        # tao = predict_tao[tech_name]
            predictions = []
            for test_file in test_files:
                for tfunc in function_in_test[test_file]:
                    test_module = test_file + "@" + tfunc
                    for production_file in production_files:
                        for pfunc in function_in_production[production_file]:
                                production_module = production_file + "@" + pfunc
                                if method_score[tech_name][test_module][production_module] > tao:
                                    predictions.append((test_file, tfunc, production_file, pfunc))
            precision, recall, f1, mAP, auc, tp, fp = util.get_metric(gts, predictions, score_alls)
            if max_f1 < f1:
                max_tao = tao
                max_precision = precision
                max_recall = recall
                max_f1 = f1
                max_mAP = mAP
                max_auc = auc
                max_tp = tp
                max_fp = fp
        print(",".join([tech_name, str(max_precision), str(max_recall), str(max_f1), str(max_mAP), str(max_auc), str(max_tp), str(max_fp), str(max_tao)]))
        # print(",".join([tech_name, str(precision), str(recall), str(f1), str(mAP), str(auc), str(tp), str(fp)]))
        
        
    # s = pname
    # ss = pname
    # x = []
    # score_all = util.get_score_all_method(scoredf_method, test)
    # for i in score_all:
    #     if i in gt:
    #         x.append(1)
    #     else:
    #         x.append(0)
    # if test_tao:
    #     for i in range(len(technique_list)):
    #         tech_name = technique_list[i]
    #         max_pre = 0
    #         max_tao = 0
    #         max_score = ""
    #         for pp in range(0, 101, 3):
    #             tao = 0.01 * pp
    #             ss = tech_name.replace("Leven", "LD")
    #             prediction = []
    #             for test_file in test_files:
    #                 for tfunc in function_in_test[test_file]:
    #                     test_module = test_file + "@" + tfunc
    #                     for production_file in production_files:
    #                         for pfunc in function_in_production[production_file]:
    #                             production_module = production_file + "@" + pfunc
    #                             if method_score[tech_name][test_module][production_module] >= tao:
    #                                 prediction.append((test_file, tfunc, production_file, pfunc))
    #             prediction_size = len(prediction)
    #             y = []
    #             for j in score_all:
    #                 if j in prediction:
    #                     y.append(1)
    #                 else:
    #                     y.append(0)
    #             tp = len(set(prediction) & set(gt))
    #             fp = prediction_size - tp
    #             fn = gt_size - tp
    #             try:
    #                 precision = tp/(tp+fp)
    #             except:
    #                 precision = 0
    #             ss = ss + " & " + str(round(precision*100, 1))
    #             recall = tp/(tp+fn)
    #             ss = ss + " & " +  str(round(recall*100, 1))
    #             try:
    #                 f1 = 2*precision*recall/(precision+recall)
    #             except:
    #                 f1 = 0
    #             ss = ss + " & " + str(round(f1*100, 1))
    #             mAP = util.cal_map(x, y)
    #             ss = ss + " & " + str(round(mAP*100, 1))
    #             try:
    #                 auc = util.cal_auc1(x, y)
    #             except:
    #                 auc = 0
    #             if i in [0, 1, 5, 8, 9]:
    #                 ss = ss + " & -"
    #             else:
    #                 ss = ss + " & " + str(round(auc*100, 1))
    #             ss = ss + "| " + str(tp) + "," + str(fp) + " \\\\\n" 
    #             if max_pre <= round(f1*100, 1):
    #                 max_pre = round(f1*100, 1)
    #                 max_tao = tao
    #                 max_score = ss
    #         print(max_tao)
    #         print(max_score)
    # else:
    #     for i in range(len(technique_list)):
    #         tech = technique_list[i]
    #         tao = predict_tao[tech]
    #         s = s + "," + tech
    #         ss = ss + " & " + tech.replace("Leven", "LD")
    #         prediction = []
    #         for test_file in test_files:
    #             for tfunc in function_in_test[test_file]:
    #                 test_module = test_file + "@" + tfunc
    #                 for production_file in production_files:
    #                     for pfunc in function_in_production[production_file]:
    #                         production_module = production_file + "@" + pfunc
    #                         if method_score[tech][test_module][production_module] >= tao:
    #                             prediction.append((test_file, tfunc, production_file, pfunc))
    #         prediction = list(set(prediction))
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
    #         mAP = util.cal_map(x, y)
    #         s = s + "," + str(round(mAP*100, 1))
    #         ss = ss + " & " + str(round(mAP*100, 1))
    #         try:
    #             auc = util.cal_auc1(x, y)
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

if __name__ == "__main__":
    pname = "face_recognition"
    proot = "F:\\myPythonProjects\\projects_new\\" + pname
    scoredf_method = pd.read_csv("score\\score_method_level_" + pname + ".csv")
    scoredf_file = pd.read_csv("score\\score_file_level_" + pname + ".csv")
    technique_list = util.get_technique_list()
    # prediction_method_to_file(technique_list)
    prediction_file_to_method(technique_list)
    