from collections import defaultdict
import pandas as pd
import random, csv, util
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

test_tao = True

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

def get_score_method(test_files, function_in_test, df, technique_list):
    result = dict()
    for tech in technique_list:
        result[tech] = defaultdict()
        for test_file in test_files:
            for func in function_in_test[test_file]:
                test_module = test_file + '@' + func
                result[tech][test_module] = defaultdict(float)
    n = df.shape[0]
    cnt = 0
    for row in df.itertuples():
        cnt+=1
        test_module = str(row.test_root) + "@" + str(row.test_method)
        production_module = str(row.production_root) + "@" + str(row.production_method)
        if not row.test_method.startswith("test_"):
            continue
        for i in range(len(technique_list)):
            tech = technique_list[i]
            try: 
                result[tech][test_module][production_module] = float(row[5+i])
            except:
                pass
    return result

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
    return score

def get_file_func(df):
    test_files = []
    production_files = []
    function_in_test = defaultdict()
    function_in_production = defaultdict()
    for row_id in range(df.shape[0]):
        test_file = df.iloc[row_id]['test_root'].strip()
        test_method = df.iloc[row_id]['test_method'].strip()
        production_file = df.iloc[row_id]['production_root'].strip()
        production_method = df.iloc[row_id]['production_method'].strip()
        test_files.append(test_file)
        production_files.append(production_file)
        if test_method.startswith("test_"):
            try:
                function_in_test[test_file].append(test_method)
            except:
                function_in_test[test_file] = [test_method]
        try:
            function_in_production[production_file].append(production_method)
        except:
            function_in_production[production_file] = [production_method]
    test_files = list(set(test_files))
    production_files = list(set(production_files))
    for i in test_files:
        function_in_test[i] = list(set(function_in_test[i]))
    for i in production_files:
        function_in_production[i] = list(set(function_in_production[i]))
    return test_files, production_files, function_in_test, function_in_production

def get_file(df):
    test_files = []
    production_files = []
    for row_id in range(df.shape[0]):
        test_files.append(df.iloc[row_id]['test_file'].strip())
        production_files.append(df.iloc[row_id]['production_file'].strip())
    test_files = list(set(test_files))
    production_files = list(set(production_files))
    return test_files, production_files

def get_links(df, tech, threshold):
    result = []
    for row_id in range(df.shape[0]):
        if df.iloc[row_id][tech] >= threshold:
            result.append((df.iloc[row_id][tech], df.iloc[row_id]['test_module'].strip(),df.iloc[row_id]['production_module'].strip()))
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
            result.append((df.iloc[row_id]['test_module'].strip(),df.iloc[row_id]['production_module'].strip()))
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

def get_all(df, level):
    result = []
    for row_id in range(df.shape[0]):
        result.append((df.iloc[row_id]['test_file' if level == "file" else 'test_module'].strip(),df.iloc[row_id]['production_file' if level == "file" else 'production_module'].strip()))
    return result

def prediction_method_simple(technique_list):
    projects_list = ["boltons", "flask", "httpie", "requests", "scrapy", "textdistance", "face_recognition"]
    pjn = len(projects_list)
    gts = []
    tests = []
    scoredf_method_list = []
    production_files = []
    print("start pre solve")
    for pj in projects_list:
        print(pj)
        proot = "projects_new\\" + pj
        scoredf_method_list.append(pd.read_csv("result\\score_method_level_" + pj + ".csv"))
        gt = util.get_truth(pj, "method")
        gts.extend(gt)
        test = get_test2(gt)
        tests.extend(test)
        pf = util.get_production_files(proot)
        production_files.extend([item.replace(util.pro_root, "") for item in pf])
    print("end pre solve")
    scoredf_method = pd.concat(scoredf_method_list)
    score_alls = util.get_score_all_method(scoredf_method, tests)
    test_files, function_in_test, _, _ = util.get_file_and_method_from_gt(gts)
    # function_in_test, test_function_sum = util.get_function_in_files(test_files, True)
    function_in_production, _ = util.get_function_in_files(production_files, False)
    method_score = get_score_method(test_files, function_in_test, scoredf_method,technique_list)
    tao = 0.5
    predictions = []
    for test_file in test_files:
        for tfunc in function_in_test[test_file]:
            test_module = test_file + "@" + tfunc
            for production_file in production_files:
                for pfunc in function_in_production[production_file]:
                    production_module = production_file + "@" + pfunc
                    score_sum = 0.0
                    for tech in technique_list:
                        score_sum += method_score[tech][test_module][production_module]
                    simple_score = score_sum / len(technique_list)
                    if simple_score > tao:
                        predictions.append((test_file, tfunc, production_file, pfunc))
    print("prediction size:", len(predictions))
    precision, recall, f1, mAP, auc, tp, fp = util.get_metric(gts, predictions, score_alls)
    print("%.1f & %.1f & %.1f & %.1f & %.1f" % (precision, recall, f1, mAP, auc))
    fp_set, fn_set = util.get_fpfn(gts, predictions)
    print(len(fp_set))
    with open("simple_method_fp.csv", "w") as f:
        f.write("test root,test method,production root,production method\n")
        for i in fp_set:
            f.write(str(i[0]) + "," + str(i[1]) + "," + str(i[2]) + "," + str(i[3]) + "\n")
    print(len(fn_set))
    with open("simple_method_fn.csv", "w") as f:
        f.write("test root,test method,production root,production method\n")
        for i in fn_set:
            f.write(str(i[0]) + "," + str(i[1]) + "," + str(i[2]) + "," + str(i[3]) + "\n")


def prediction_file_simple(technique_list):
    projects_list = ["boltons", "flask", "httpie", "requests", "scrapy", "textdistance", "face_recognition"]
    pjn = len(projects_list)
    gts = []
    tests = []
    test_files = []
    scoredf_file_list = []
    production_files = []
    print("start pre solve")
    for pj in projects_list:
        print(pj)
        scoredf_file_list.append(pd.read_csv("result\\score_file_level_" + pj + ".csv"))
        gt = util.get_truth(pj, "module")
        gts.extend(gt)
        test = get_test1(gt)
        tests.extend(test)
        tf = util.get_files_from_gt(gt, 0)
        test_files.extend(tf)
        pf = util.get_files_from_gt(gt, 1)
        production_files.extend(pf)
    print("end pre solve")
    scoredf_file = pd.concat(scoredf_file_list)
    score_alls = util.get_score_all_file(scoredf_file, tests)
    file_score = get_score_file(test_files, production_files, scoredf_file, technique_list)
    tao = 0.4
    predictions = []
    for test_file in test_files:
        for production_file in production_files:
            score_sum = 0.0
            for tech in technique_list:
                score_sum += file_score[tech][test_file][production_file]
            simple_score = score_sum / len(technique_list)
            if simple_score > tao:
                predictions.append((test_file, production_file))
    # print("prediction size:", len(predictions))
    precision, recall, f1, mAP, auc, tp, fp = util.get_metric(gts, predictions, score_alls)
    print("%.1f & %.1f & %.1f & %.1f & %.1f" % (precision, recall, f1, mAP, auc))
    fp_set, fn_set = util.get_fpfn(gts, predictions)
    print(len(fp_set), fp_set)
    with open("simple_module_fp.csv", "w") as f:
        f.write("test module,production module\n")
        for i in fp_set:
            f.write(str(i[0]) + "," + str(i[1]) + "\n")
    print(len(fn_set), fn_set)
    with open("simple_module_fn.csv", "w") as f:
        f.write("test module,production module\n")
        for i in fn_set:
            f.write(str(i[0]) + "," + str(i[1]) + "\n")


def prediction_method_weight(technique_list):
    weight = [97.9, 62.8, 27.5, 18.6, 19.3, 13.0, 7.3, 0.1, 100.0, 72.4, 1.9, 1.6, 2.1, 0.2, 0.4]
    ss = 0
    for i in weight:
        ss += i
    for i in range(len(weight)):
        weight[i] = weight[i] / ss
    projects_list = ["boltons", "flask", "httpie", "requests", "scrapy", "textdistance", "face_recognition"]
    pjn = len(projects_list)
    gts = []
    tests = []
    scoredf_method_list = []
    production_files = []
    print("start pre solve")
    for pj in projects_list:
        print(pj)
        proot = "projects_new\\" + pj
        scoredf_method_list.append(pd.read_csv("result\\score_method_level_" + pj + ".csv"))
        gt = util.get_truth(pj, "method")
        gts.extend(gt)
        test = get_test2(gt)
        tests.extend(test)
        pf = util.get_production_files(proot)
        production_files.extend([item.replace(util.pro_root, "") for item in pf])
    print("end pre solve")
    scoredf_method = pd.concat(scoredf_method_list)
    score_alls = util.get_score_all_method(scoredf_method, tests)
    test_files, function_in_test, _, _ = util.get_file_and_method_from_gt(gts)
    # function_in_test, test_function_sum = util.get_function_in_files(test_files, True)
    function_in_production, _ = util.get_function_in_files(production_files, False)
    method_score = get_score_method(test_files, function_in_test, scoredf_method,technique_list)
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
        print(tao)
        predictions = []
        for test_file in test_files:
            for tfunc in function_in_test[test_file]:
                test_module = test_file + "@" + tfunc
                for production_file in production_files:
                    for pfunc in function_in_production[production_file]:
                        production_module = production_file + "@" + pfunc
                        score_weight = 0.0
                        for i in range(len(technique_list)):
                            tech = technique_list[i]
                            score_weight += weight[i] * method_score[tech][test_module][production_module]
                        if score_weight > tao:
                            predictions.append((test_file, tfunc, production_file, pfunc))
        print("prediction size:", len(predictions))
        precision, recall, f1, mAP, auc, tp, fp = util.get_metric(gts, predictions, score_alls)
        print("%.1f & %.1f & %.1f & %.1f & %.1f" % (precision, recall, f1, mAP, auc))
        if max_f1 < f1:
            max_tao = tao
            max_precision = precision
            max_recall = recall
            max_f1 = f1
            max_mAP = mAP
            max_auc = auc
            max_tp = tp
            max_fp = fp
    print(",".join([str(max_precision), str(max_recall), str(max_f1), str(max_mAP), str(max_auc), str(max_tp), str(max_fp), str(max_tao)]))


def prediction_file_weight(technique_list):
    weight = [96.1, 75.2, 87.0, 90.7, 97.5, 13.8, 4.9, 2.8, 91.1, 64.5, 13.7, 17.0, 22.2, 1.4, 1.6]
    ss = 0
    for i in weight:
        ss += i
    for i in range(len(weight)):
        weight[i] = weight[i] / ss
    projects_list = ["boltons", "flask", "httpie", "requests", "scrapy", "textdistance", "face_recognition"]
    pjn = len(projects_list)
    gts = []
    tests = []
    test_files = []
    scoredf_file_list = []
    production_files = []
    print("start pre solve")
    for pj in projects_list:
        print(pj)
        scoredf_file_list.append(pd.read_csv("result\\score_file_level_" + pj + ".csv"))
        gt = util.get_truth(pj, "module")
        gts.extend(gt)
        test = get_test1(gt)
        tests.extend(test)
        tf = util.get_files_from_gt(gt, 0)
        test_files.extend(tf)
        pf = util.get_files_from_gt(gt, 1)
        production_files.extend(pf)
    print("end pre solve")
    scoredf_file = pd.concat(scoredf_file_list)
    score_alls = util.get_score_all_file(scoredf_file, tests)
    file_score = get_score_file(test_files, production_files, scoredf_file, technique_list)
    max_tao = 0
    max_precision = 0
    max_recall = 0
    max_f1 = 0
    max_mAP = 0
    max_auc = 0
    for pp in range(0, 101, 10):
        tao = 0.01 * pp
        print(tao)
        predictions = []
        for test_file in test_files:
            for production_file in production_files:
                score_weight = 0.0
                for i in range(len(technique_list)):
                    tech = technique_list[i]
                    score_weight += weight[i] * file_score[tech][test_file][production_file]
                if score_weight > tao:
                    predictions.append((test_file, production_file))
        # print("prediction size:", len(predictions))
        precision, recall, f1, mAP, auc, tp, fp = util.get_metric(gts, predictions, score_alls)
        # print(",".join([str(precision), str(recall), str(f1), str(mAP), str(auc), str(tp), str(fp)]))
        if max_f1 < f1:
            max_tao = tao
            max_precision = precision
            max_recall = recall
            max_f1 = f1
            max_mAP = mAP
            max_auc = auc
    print("%.1f & %.1f & %.1f & %.1f & %.1f" % (max_precision, max_recall, max_f1, max_mAP, max_auc))


def prediction_method_exception(technique_list):
    projects_list = ["boltons", "flask", "httpie", "requests", "scrapy", "textdistance", "face_recognition"]
    pjn = len(projects_list)
    gts = []
    tests = []
    scoredf_method_list = []
    production_files = []
    print("start pre solve")
    for pj in projects_list:
        print(pj)
        proot = "projects_new\\" + pj
        scoredf_method_list.append(pd.read_csv("result\\score_method_level_" + pj + ".csv"))
        gt = util.get_truth(pj, "method")
        gts.extend(gt)
        test = get_test2(gt)
        tests.extend(test)
        pf = util.get_production_files(proot)
        production_files.extend([item.replace(util.pro_root, "") for item in pf])
    print("end pre solve")
    scoredf_method = pd.concat(scoredf_method_list)
    score_alls = util.get_score_all_method(scoredf_method, tests)
    test_files, function_in_test, _, _ = util.get_file_and_method_from_gt(gts)
    # function_in_test, test_function_sum = util.get_function_in_files(test_files, True)
    function_in_production, _ = util.get_function_in_files(production_files, False)
    method_score = get_score_method(test_files, function_in_test, scoredf_method,technique_list)
    for excep_tech in technique_list:
        # print(excep_tech)
        max_tao = 0
        max_precision = 0
        max_recall = 0
        max_f1 = 0
        max_mAP = 0
        max_auc = 0
        max_tp = 0
        max_fp = 0
        for pp in range(10, 91, 10):
            tao = 0.01 * pp
            # print(tao)
            predictions = []
            for test_file in test_files:
                for tfunc in function_in_test[test_file]:
                    test_module = test_file + "@" + tfunc
                    for production_file in production_files:
                        for pfunc in function_in_production[production_file]:
                            production_module = production_file + "@" + pfunc
                            score_exception = 0.0
                            for i in range(len(technique_list)):
                                tech = technique_list[i]
                                if tech == excep_tech:
                                    continue
                                score_exception += method_score[tech][test_module][production_module]
                            score_exception /= len(technique_list) - 1
                            if score_exception > tao:
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
        print("%s %.1f & %.1f & %.1f & %.1f & %.1f" % (excep_tech, max_precision, max_recall, max_f1, max_mAP, max_auc))


def prediction_file_exception(technique_list):
    projects_list = ["boltons", "flask", "httpie", "requests", "scrapy", "textdistance", "face_recognition"]
    pjn = len(projects_list)
    gts = []
    tests = []
    test_files = []
    scoredf_file_list = []
    production_files = []
    print("start pre solve")
    for pj in projects_list:
        print(pj)
        scoredf_file_list.append(pd.read_csv("result\\score_file_level_" + pj + ".csv"))
        gt = util.get_truth(pj, "module")
        gts.extend(gt)
        test = get_test1(gt)
        tests.extend(test)
        tf = util.get_files_from_gt(gt, 0)
        test_files.extend(tf)
        pf = util.get_files_from_gt(gt, 1)
        production_files.extend(pf)
    print("end pre solve")
    scoredf_file = pd.concat(scoredf_file_list)
    score_alls = util.get_score_all_file(scoredf_file, tests)
    file_score = get_score_file(test_files, production_files, scoredf_file, technique_list)
    for excep_tech in technique_list:
        max_tao = 0
        max_precision = 0
        max_recall = 0
        max_f1 = 0
        max_mAP = 0
        max_auc = 0
        max_tp = 0
        max_fp = 0
        for pp in range(0, 101, 10):
            tao = 0.01 * pp
            # print(tao)
            predictions = []
            for test_file in test_files:
                for production_file in production_files:
                    score_exception = 0.0
                    for i in range(len(technique_list)):
                        tech = technique_list[i]
                        if tech == excep_tech:
                            continue
                        score_exception += file_score[tech][test_file][production_file]
                    score_exception /= len(technique_list) - 1
                    if score_exception > tao:
                        predictions.append((test_file, production_file))
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
        print("%s %.1f & %.1f & %.1f & %.1f & %.1f" % (excep_tech, max_precision, max_recall, max_f1, max_mAP, max_auc))


def get_train(level, technique_list):
    tx = []
    ty = []
    cnt = 0
    l = ["boltons", "flask", "httpie", "requests", "scrapy", "textdistance", "face_recognition"]
    dfs = []
    for pname in l:
        dfs.append(pd.read_csv("score\\score_" + level + "_level_" + pname + ".csv"))
    df = pd.concat([dfs[0], dfs[1], dfs[2], dfs[3], dfs[4], dfs[5], dfs[6]], ignore_index=True)
    df["sum"] = 0
    for row_id in range(df.shape[0]):
        su = 0
        for tech in technique_list:
            su += df.iloc[row_id][tech]
        df.loc[row_id, "sum"] = su
        if df.iloc[row_id]["NC"] == 1:
            cnt = cnt + 1
            tmp = []
            for tech in technique_list:
                tmp.append(df.iloc[row_id][tech])
            tx.append(tmp)
            ty.append(1)
    df = df.sort_values(by="sum", ascending=True)
    for i in range(cnt):
        tmp = []
        for tech in technique_list:
            tmp.append(df.iloc[i][tech])
        tx.append(tmp)
        ty.append(0)
    return tx, ty

def get_train_method(gt, df, technique_list):
    tx = []
    ty = []
    cnt = 0
    for row in df.itertuples():
        if (str(row.test_root), str(row.test_method), str(row.production_root), str(row.production_method)) in gt:
            cnt = cnt + 1
            tmp = []
            for i in range(len(technique_list)):
                tmp.append(float(row[5+i]))
            tx.append(tmp)
            ty.append(1)
    for row in df.itertuples():
        if (str(row.test_root), str(row.test_method), str(row.production_root), str(row.production_method)) not in gt and cnt > 0:
            cnt = cnt - 1
            tmp = []
            for i in range(len(technique_list)):
                tmp.append(float(row[5+i]))
            tx.append(tmp)
            ty.append(0)
        if cnt <= 0:
            break
    return tx, ty

def get_train_file(gt, df, technique_list):
    tx = []
    ty = []
    cnt = 0
    for row in df.itertuples():
        if (str(row.test_file), str(row.production_file)) in gt:
            cnt = cnt + 1
            tmp = []
            for i in range(len(technique_list)):
                tmp.append(float(row[3+i]))
            tx.append(tmp)
            ty.append(1)
    for row in df.itertuples():
        if (str(row.test_file), str(row.production_file)) not in gt and cnt > 0:
            cnt = cnt - 1
            tmp = []
            for i in range(len(technique_list)):
                tmp.append(float(row[3+i]))
            tx.append(tmp)
            ty.append(0)
        if cnt <= 0:
            break
    return tx, ty

def get_test_method(df, technique_list, tests):
    te = []
    result = []
    for row in df.itertuples():
        if (str(row.test_root), str(row.test_method)) in tests:
            tmp = []
            for i in range(len(technique_list)):
                tmp.append(float(row[5+i]))
            te.append(tmp)
            result.append((str(row.test_root), str(row.test_method), str(row.production_root), str(row.production_method)))
    return te, result

def get_test_file(df, technique_list, tests):
    te = []
    result = []
    for row in df.itertuples():
        if str(row.test_file) in tests:
            tmp = []
            for i in range(len(technique_list)):
                tmp.append(float(row[3+i]))
            te.append(tmp)
            result.append((str(row.test_file), str(row.production_file)))
    return te, result

def prediction_method_ml(technique_list):
    projects_list = ["boltons", "flask", "httpie", "requests", "scrapy", "textdistance", "face_recognition"]
    gts = []
    tests = []
    scoredf_method_list = []
    print("start pre solve")
    for pj in projects_list:
        print(pj)
        scoredf_method_list.append(pd.read_csv("result\\score_method_level_" + pj + ".csv"))
        gt = util.get_truth(pj, "method")
        gts.extend(gt)
        test = get_test2(gt)
        tests.extend(test)

    print("end pre solve")
    scoredf_method = pd.concat(scoredf_method_list)
    score_alls = util.get_score_all_method(scoredf_method, tests)
    train_x, train_y = get_train_method(gts, scoredf_method, technique_list)
    test_x, pp = get_test_method(scoredf_method, technique_list, tests)
    lr = LogisticRegression()
    lr.fit(train_x, train_y)
    predi = lr.predict(test_x)
    predictions = []
    for i in range(len(test_x)):
        if predi[i] == 1:
            predictions.append(pp[i])
    precision, recall, f1, mAP, auc, tp, fp = util.get_metric(gts, predictions, score_alls)
    print(",".join([str(precision), str(recall), str(f1), str(mAP), str(auc), str(tp), str(fp)]))
    print("%.1f & %.1f & %.1f & %.1f & %.1f" % (precision, recall, f1, mAP, auc))


def prediction_file_ml(technique_list):
    projects_list = ["boltons", "flask", "httpie", "requests", "scrapy", "textdistance", "face_recognition"]
    pjn = len(projects_list)
    gts = []
    tests = []
    test_files = []
    scoredf_file_list = []
    production_files = []
    print("start pre solve")
    for pj in projects_list:
        print(pj)
        scoredf_file_list.append(pd.read_csv("result\\score_file_level_" + pj + ".csv"))
        gt = util.get_truth(pj, "module")
        gts.extend(gt)
        test = get_test1(gt)
        tests.extend(test)
        tf = util.get_files_from_gt(gt, 0)
        test_files.extend(tf)
        pf = util.get_files_from_gt(gt, 1)
        production_files.extend(pf)
    print("end pre solve")
    
    scoredf_file = pd.concat(scoredf_file_list)
    score_alls = util.get_score_all_file(scoredf_file, tests)
    train_x, train_y = get_train_file(gts, scoredf_file, technique_list)
    test_x, pp = get_test_file(scoredf_file, technique_list, tests)
    print(len(train_x))
    lr = LogisticRegression()
    lr.fit(train_x, train_y)
    predi = lr.predict(test_x)
    predictions = []
    for i in range(len(test_x)):
        if predi[i] == 1:
            predictions.append(pp[i])
    print(len(predictions), len(score_alls))
    precision, recall, f1, mAP, auc, tp, fp = util.get_metric(gts, predictions, score_alls)
    print("%.1f & %.1f & %.1f & %.1f & %.1f" % (precision, recall, f1, mAP, auc))


def prediction_file_new(project_root, pname, scoredf_file, technique_list):
    predict_tao = { "LCS-U"         : 42.7,\
                    "LCS-B"         : 42.3,\
                    "Leven"         : 50.7,\
                    "LCBA"          : 45.4,\
                    "Tarantula"     : 35.3,\
                    "TFIDF"         : 15.6,\
                    "Static LCS-U"  : 17.4,\
                    "Static LCS-B"  : 31.6,\
                    "Static Leven"  : 50.3}
    tao_sum = 0
    for key, value in predict_tao.items():
        tao_sum += value
    for key, value in predict_tao.items():
        predict_tao[key] /= tao_sum
    
    if test_tao:
        max_precision = 0.0
        max_recall = 0.0
        max_f1 = 0.0
        max_tao = 0.0
        for pp in range(60, 101, 1):
            f1_sum = 0.0
            precision_sum = 0.0
            recall_sum = 0.0
            taoo = 0.01 * pp
            for pname in ["boltons", "flask", "httpie", "requests", "scrapy", "textdistance", "face_recognition"]:
                project_root = util.pro_root + "projects_new\\" + pname
                scoredf_file = pd.read_csv("score\\score_file_level_" + pname + ".csv")
                file_score = get_score_file(project_root,scoredf_file,technique_list)
                test_files = util.get_test_files(project_root)
                production_files = util.get_production_files(project_root)
                # new_score = dict()
                # for tfile in test_files:
                #     new_score[tfile] = dict()
                #     for pfile in production_files:
                #         new_score[tfile][pfile] = 0.0
                prediction = []
                for test_file in test_files:
                    for production_file in production_files:
                        if file_score["NC"][test_file][production_file] >= 0.5 or \
                        file_score["NCC"][test_file][production_file] >= 0.5 or \
                        file_score["Static NC"][test_file][production_file] >= 0.5 or \
                        file_score["Static NCC"][test_file][production_file] >= 0.5:
                            prediction.append((test_file, production_file))
                        else:
                            score_sum = 0.0
                            for tech, tao in predict_tao.items():
                                score_sum += file_score[tech][test_file][production_file]
                            score_sum /= len(predict_tao)
                            # new_score[test_file][production_file] = score_sum
                            if score_sum > taoo:
                                prediction.append((test_file, production_file))
                x = []
                gt = util.get_truth(pname, "module")
                gt_size = len(gt)
                test = get_test1(gt)
                score_all = util.get_score_all_file(scoredf_file, test)
                for i in score_all:
                    if i in gt:
                        x.append(1)
                    else:
                        x.append(0)
                prediction = list(set(prediction))
                prediction_size = len(prediction)
                y = []
                for j in score_all:
                    if j in prediction:
                        y.append(1)
                    else:
                        y.append(0)
                tp = len(set(prediction) & set(gt))
                fp = prediction_size - tp
                fn = gt_size - tp
                try: precision = tp/(tp+fp)
                except: precision = 0
                try: recall = tp/(tp+fn)
                except: recall = 0
                try: f1 = 2*precision*recall/(precision+recall)
                except: f1 = 0
                # print(pname, precision, recall, f1)
                precision_sum += precision
                recall_sum += recall
                f1_sum += f1
            if max_f1 <= f1_sum / 7.0:
                max_f1 = f1_sum / 7.0
                max_precision = precision_sum / 7.0
                max_recall = recall_sum / 7.0
                max_tao = taoo
            print(taoo, max_tao, max_precision, max_recall, max_f1)
        print(max_tao, max_precision)
    else:
        scoredf_file = pd.read_csv("score\\score_file_level_" + pname + ".csv")
        file_score = get_score_file(project_root,scoredf_file,technique_list)
        test_files = util.get_test_files(project_root)
        production_files = util.get_production_files(project_root)
        prediction = []
        for test_file in test_files:
            for production_file in production_files:
                if file_score["NC"][test_file][production_file] >= 0.5 or \
                file_score["NCC"][test_file][production_file] >= 0.5 or \
                file_score["Static NC"][test_file][production_file] >= 0.5 or \
                file_score["Static NCC"][test_file][production_file] >= 0.5:
                    prediction.append((test_file, production_file))
                else:
                    score_sum = 0.0
                    for tech, tao in predict_tao.items():
                        score_sum += file_score[tech][test_file][production_file]
                    score_sum /= len(predict_tao)
                    if score_sum > 0.66:
                        prediction.append((test_file, production_file))
                        pass
        s = pname
        ss = pname
        x = []
        gt = util.get_truth(pname, "module")
        gt_size = len(gt)
        test = get_test1(gt)
        score_all = util.get_score_all_file(scoredf_file, test)
        for i in score_all:
            if i in gt:
                x.append(1)
            else:
                x.append(0)
        prediction = list(set(prediction))
        prediction_size = len(prediction)
        y = []
        for j in score_all:
            if j in prediction:
                y.append(1)
            else:
                y.append(0)
        tp = len(set(prediction) & set(gt))
        fp = prediction_size - tp
        fn = gt_size - tp
        try:
            precision = tp/(tp+fp)
        except:
            precision = 0
        s = s + "," + str(round(precision*100, 1))
        ss = ss + " & " + str(round(precision*100, 1))
        recall = tp/(tp+fn)
        s = s + "," + str(round(recall*100, 1))
        ss = ss + " & " +  str(round(recall*100, 1))
        try:
            f1 = 2*precision*recall/(precision+recall)
        except:
            f1 = 0
        s = s + "," + str(round(f1*100, 1))
        ss = ss + " & " + str(round(f1*100, 1))
        mAP = cal_map(x, y)
        s = s + "," + str(round(mAP*100, 1))
        ss = ss + " & " + str(round(mAP*100, 1))
        try:
            auc = cal_auc1(x, y)
        except:
            auc = 0
        s = s + "," + str(round(auc*100, 1))
        ss = ss + " & " + str(round(auc*100, 1))
        s = s + "," + str(tp) + "," + str(fp)  + "\n"
        ss = ss + " \\\\\n" + str(tp) + " " +str(fp)
        print(ss)
        with open("res.csv", "w") as f:
            f.write(s)

def prediction_file_newnew(project_root, pname, scoredf_file, technique_list):
    predict_tao = { "LCS-U"         : 42.7,\
                    "LCS-B"         : 42.3,\
                    "Leven"         : 50.7,\
                    "LCBA"          : 45.4,\
                    "Tarantula"     : 35.3,\
                    "TFIDF"         : 15.6,\
                    "Static LCS-U"  : 17.4,\
                    "Static LCS-B"  : 31.6,\
                    "Static Leven"  : 50.3}
    weight = [66.8, 59.1, 42.7, 42.3, 50.7, 45.4, 35.3, 15.6, 64.4, 57.4, 17.4, 31.6, 50.3]
    ss = 0
    for i in weight:
        ss += i
    for i in range(len(weight)):
        weight[i] = weight[i] / ss
    tao_sum = 0
    for key, value in predict_tao.items():
        tao_sum += value
    for key, value in predict_tao.items():
        predict_tao[key] /= tao_sum
    precision_map = dict()
    recall_map = dict()
    f1_map = dict()
    for pp in range(0, 101, 1):
        tau = 0.01 * pp
        precision_map[tau] = 0.0
        recall_map[tau] = 0.0
        f1_map[tau] = 0.0
    for pp in range(0, 101, 1):
        taoo = 0.01 * pp
        f1_sum = 0.0
        precision_sum = 0.0
        recall_sum = 0.0
        for pname in ["boltons", "flask", "httpie", "requests", "scrapy", "textdistance", "face_recognition"]:
            project_root = "projects_new\\" + pname
            scoredf_file = pd.read_csv("score\\score_file_level_" + pname + ".csv")
            file_score = get_score_file(project_root,scoredf_file,technique_list)
            test_files = util.get_test_files(project_root)
            production_files = util.get_production_files(project_root)
            prediction = []
            for test_file in test_files:
                for production_file in production_files:
                    # if file_score["NC"][test_file][production_file] >= 0.5 or \
                    # file_score["NCC"][test_file][production_file] >= 0.5 or \
                    # file_score["Static NC"][test_file][production_file] >= 0.5 or \
                    # file_score["Static NCC"][test_file][production_file] >= 0.5:
                    #     prediction.append((test_file, production_file))
                    # else:
                    score_sum = 0.0
                    # for tech, tao in predict_tao.items():
                    #     score_sum += file_score[tech][test_file][production_file]
                    # for tech in technique_list:
                    #     score_sum += file_score[tech][test_file][production_file]
                    for i in range(len(weight)):
                        tech = technique_list[i]
                        score_sum += file_score[tech][test_file][production_file] * weight[i]
                    # score_sum /= len(predict_tao)
                    # score_sum /= len(technique_list)
                    if score_sum > taoo:
                        prediction.append((test_file, production_file))
            x = []
            gt = util.get_truth(pname, "module")
            gt_size = len(gt)
            test = get_test1(gt)
            score_all = util.get_score_all_file(scoredf_file, test)
            for i in score_all:
                if i in gt:
                    x.append(1)
                else:
                    x.append(0)
            prediction = list(set(prediction))
            prediction_size = len(prediction)
            y = []
            for j in score_all:
                if j in prediction:
                    y.append(1)
                else:
                    y.append(0)
            tp = len(set(prediction) & set(gt))
            fp = prediction_size - tp
            fn = gt_size - tp
            try: precision = tp/(tp+fp)
            except: precision = 0
            try: recall = tp/(tp+fn)
            except: recall = 0
            try: f1 = 2*precision*recall/(precision+recall)
            except: f1 = 0
            # print(pname, precision, recall, f1)
            precision_sum += precision
            recall_sum += recall
            f1_sum += f1
        precision_map[taoo] = precision_sum / 7.0
        recall_map[taoo] = recall_sum / 7.0
        f1_map[taoo] = f1_sum / 7.0
        print(taoo, precision_map[taoo], recall_map[taoo], f1_map[taoo])
    with open("tau_metrics_weight.csv", "w") as f:
        f.write("tau,precision,recall,f1-score\n")
        for pp in range(0, 101, 1):
            tau = 0.01 * pp
            f.write(",".join([str(tau), str(precision_map[tau]), str(recall_map[tau]), str(f1_map[tau])]) + "\n")


if __name__ == "__main__":
    technique_list = util.get_technique_list()
    prediction_method_simple(technique_list)
    prediction_file_simple(technique_list)
    prediction_method_weight(technique_list)
    prediction_file_weight(technique_list)
    prediction_method_exception(technique_list)
    prediction_file_exception(technique_list)
    prediction_method_ml(technique_list)
    prediction_file_ml(technique_list)