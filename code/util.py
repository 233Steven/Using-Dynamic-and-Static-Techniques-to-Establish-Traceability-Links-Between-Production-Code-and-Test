from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
import os, re, random, difflib, git, xlrd
import numpy as np

call_path = None
pro_root = "Please set your project root path here!"

def get_technique_list():
    return ["NC", "NCC", "LCS-U", "LCS-B", "Leven", "LCBA", "Tarantula", "TFIDF", \
            "Static NC", "Static NCC", "Static LCS-U", "Static LCS-B", "Static Leven", "Similarity", "Co-ev"]

def write_log(x, mode="w"):
    with open("log.txt", mode=mode) as f:
        f.write(str(x))
        f.write("\n\n")

def get_truth(pname, level="method"):
    result = []
    wb = xlrd.open_workbook("groundtruth/" + level + "-all.xls")
    sheet = wb.sheet_by_name("final")
    nrows =sheet.nrows
    ncols = sheet.ncols
    for i in range(nrows):
        if "\\" + pname + "\\" in sheet.cell_value(i, 0):
            tmp = []
            for j in range(ncols):
                tmp.append(sheet.cell_value(i, j))
            result.append(tuple(tmp))
    return result

def get_prediction_method(df, tech, threshold, test):
    result = []
    for row_id in range(df.shape[0]):
        if (df.iloc[row_id]['test_root'].strip(),df.iloc[row_id]['test_method'].strip()) in test:
            if df.iloc[row_id][tech] >= threshold:
                result.append((df.iloc[row_id][tech], df.iloc[row_id]['test_root'].strip(),df.iloc[row_id]['test_method'].strip(), \
                        df.iloc[row_id]['production_root'].strip(),df.iloc[row_id]['production_method'].strip()))
    result.sort()
    tmp = []
    for i in result:
        tmp.append((i[1], i[2], i[3], i[4]))
    return tmp

def get_prediction_file(df, tech, threshold, test):
    result = []
    for row in df.itertuples():
        if row.test_file in test and row[tech] > threshold:
            result.append((row.test_file, row.production_file))
    # for row_id in range(df.shape[0]):
    #     if df.iloc[row_id]['test_file'].strip() in test:
    #         if df.iloc[row_id][tech] >= threshold:
    #             result.append((df.iloc[row_id][tech], df.iloc[row_id]['test_file'].strip(),df.iloc[row_id]['production_file'].strip()))
    result.sort()
    tmp = []
    for i in result:
        tmp.append((i[1], i[2]))
    return tmp

def get_score_all_method(df, test):
    result = []
    for row in df.itertuples():
        if (row.test_root, row.test_method) in test:
            result.append((row.test_root, row.test_method, row.production_root, row.production_method))
    # for row_id in range(df.shape[0]):
    #     if (df.iloc[row_id]['test_root'].strip(),df.iloc[row_id]['test_method'].strip()) in test:
    #         result.append((df.iloc[row_id]['test_root'].strip(),df.iloc[row_id]['test_method'].strip(), \
    #                         df.iloc[row_id]['production_root'].strip(),df.iloc[row_id]['production_method'].strip()))
    return result

def get_score_all_file(df, test):
    result = []
    for row in df.itertuples():
        if row.test_file in test:
            result.append((row.test_file, row.production_file))
    # for row_id in range(df.shape[0]):
    #         if df.iloc[row_id]['test_file'].strip() in test:
    #             result.append((df.iloc[row_id]['test_file'].strip(),df.iloc[row_id]['production_file'].strip()))
    return result

def cal_auc1(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def cal_map(y_true, y_pred):
    return average_precision_score(y_true, y_pred)

def get_fpfn(gt, prediction):
    tp_set = set(prediction) & set(gt)
    fp_set = set(prediction) - tp_set
    fn_set = set(gt) - tp_set
    return fp_set, fn_set

def get_metric(gt, prediction, score_all):
    gt_size = len(gt)
    prediction_size = len(prediction)
    x = []
    y = []
    for i in score_all:
        x.append(int(i in gt))
        y.append(int(i in prediction))
    tp = len(set(prediction) & set(gt))
    fp = prediction_size - tp
    # print("................TP...............")
    # for item in set(prediction) & set(gt):
    #     print(item)
    # print("................FP...............")
    # for item in set(prediction):
    #     if item not in set(gt):
    #         print(item)
    # print("................FN...............")
    # for item in set(gt):
    #     if item not in set(prediction):
    #         print(item)
    if prediction_size > 0: precision = tp / prediction_size * 100
    else: precision = 0
    if gt_size > 0: recall = tp / gt_size * 100
    else: recall = 0
    if precision + recall > 0: f1 = 2 * precision * recall / (precision + recall)
    else: f1 = 0
    try: mAP = cal_map(x, y) * 100
    except: mAP = 0
    try: auc = cal_auc1(x, y) * 100
    except: auc = 0
    return precision, recall, f1, mAP, auc, tp, fp

def get_files(root):
    que = list()
    que.append(root)
    result = []
    while que:
        now = que[0]
        que.pop(0)
        files = os.listdir(now)
        for f in files:
            f_d = os.path.join(now, f)
            if os.path.isfile(f_d) and f.endswith('.py'):
                result.append(f_d)
            if os.path.isdir(f_d):
                que.append(f_d)
    return result

def get_test_files(root):
    files = get_files(root)
    result = [file for file in files if file.split("\\")[-1].startswith("test_")]
    return result

def get_file_and_method_from_gt(gt):
    tfs = set()
    tt = defaultdict(set)
    pfs = set()
    pp = defaultdict(set)
    for gtt in gt:
        tfs.add(gtt[0])
        pfs.add(gtt[2])
        tt[gtt[0]].add(gtt[1])
        pp[gtt[2]].add(gtt[3])
    return tfs, tt, pfs, pp

def get_files_from_gt(gt, ind):
    result = set()
    for gtt in gt:
        result.add(gtt[ind])
    return list(result)

def get_production_files(root):
    files = get_files(root)
    result = [file for file in files if not file.split("\\")[-1].startswith("test_")]
    return result

def calculate_similarity(code1, code2):
    words1 = code1.split()
    words2 = code2.split()

    similarity = difflib.SequenceMatcher(None, words1, words2).ratio()
    if similarity > 1.0:
        print("????????????????????????????????????????????????????")
    return similarity

def get_code_in_files(test_files, production_files):
    result = defaultdict()
    for file in test_files:
        with open(file, "r", encoding="utf-8") as f:
            result[file] = f.read()
    for file in production_files:
        with open(file, "r", encoding="utf-8") as f:
            result[file] = f.read()
    return result

def get_code_in_functions(test_files, t_function_in_files, production_files, p_function_in_files):
    result = defaultdict()
    for file in test_files:
        result[file] = dict()
        with open(file, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
            l = len(lines)
            for func in t_function_in_files[file]:
                for i in range(l):
                    line = lines[i]
                    if re.match("\s*def\s+" + func + ".*", line):
                        s = len(lines[i + 1]) - len(lines[i + 1].lstrip())
                        j = i + 1
                        while j < l and (len(lines[j]) - len(lines[j].lstrip()) >= s or len(lines[j].strip()) == 0):
                            j = j + 1
                        result[file][func] = "\n".join(lines[i: j])
    for file in production_files:
        result[file] = dict()
        with open(file, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
            l = len(lines)
            for func in p_function_in_files[file]:
                for i in range(l):
                    line = lines[i]
                    if re.match("\s*def\s+" + func + ".*", line):
                        s = len(lines[i + 1]) - len(lines[i + 1].lstrip())
                        j = i + 1
                        while j < l and (len(lines[j]) - len(lines[j].lstrip()) >= s or len(lines[j].strip()) == 0):
                            j = j + 1
                        result[file][func] = "\n".join(lines[i: j])
    return result

def get_function_in_files(files, istest=False):
    result = defaultdict(set)
    function_sum = 0
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
            for line in lines:
                if istest:
                    if re.match("\s*def\s+test_.*", line):
                        function_name = line.split()[1].split("(")[0].strip()
                        result[file].add(function_name)
                else:
                    if re.match("\s*def\s+.*", line):
                        function_name = line.split()[1].split("(")[0].strip()
                        # if not function_name.startswith("__"):
                        result[file].add(function_name)
        result[file] = list(result[file])
        function_sum += len(result[file])
    return result, function_sum

def get_git_info_method(gitRoot):
    print("start cal git")
    repo = git.Repo(gitRoot)
    commits = repo.iter_commits()
    result = []
    # cmts = []
    for c in commits:
        if not c.parents:
            continue
        else:
            parentCommit = c.parents[0]
            diffs = parentCommit.diff(c, create_patch=True)
            tmp = set()
            # l = ""
            for diff in diffs.iter_change_type("M"):
                if diff.a_path and diff.a_path != diff.b_path:
                    continue
                file = diff.a_path
                if not file.endswith(".py"):
                    continue
                file_name = file.replace(".py", "").replace("/", ".").replace("\\", ".").split(".")[-1]
                lines = str(diff).split("\n")
                # l = l + str(diff) + "\n"
                for line in lines:
                    try:
                        if re.match(".*def\s+.*", line):
                            function_name = line.split("def")[1].split("(")[0].strip()
                            # print(line, "!!!!", function_name)
                            tmp.add("@".join([file_name, function_name]))
                    except:
                        print(line)
            result.append(list(tmp))
            # cmts.append(c.hexsha)
    # with open("log.txt", "w") as f:
    #     f.write(str(cmts[101]))
    #     f.write("\n")
    #     f.write(str(result[101]))
    print("end cal git")
    return result
        

def get_git_info_file(gitRoot):
    print("start cal git")
    repo = git.Repo(gitRoot)
    commits = repo.iter_commits()
    result = []
    for c in commits:
        if not c.parents:
            continue
        else:
            stats = c.stats
            files = stats.files
            tmp = []
            for file in files.keys():
                if file.endswith(".py"):
                    file_name = file.replace(".py", "").replace("/", ".").replace("\\", ".").split(".")[-1]
                    tmp.append(file_name)
            if len(tmp) > 0:
                result.append(tmp)
    print(len(result))
    print("end cal git")
    return result

def cal_appearance_in_git(t, p, ls):
    cnt = 0
    for l in ls:
        if t in l and p in l:
            cnt = cnt + 1
    return cnt

def lcs(s1, s2):
    n = len(s1)
    m = len(s2)
    c = [[0]*(m+1) for i in range(n+1)]
    for i in range(n+1):
        for j in range(m+1):
            if i==0 or j==0:
                c[i][j] = 0
            elif s1[i-1]==s2[j-1]:
                    c[i][j] = c[i-1][j-1] + 1
            else:
                    c[i][j] = max(c[i-1][j], c[i][j-1])
    return c[n][m]

def cal1(project_list, level): # method 355  file 214
    oo = 0
    if level == "method":
        oo = 355
    else:
        oo = 214
    csv_root = "result\\"
    gt_root = "gt\\"
    d = defaultdict(int)
    p = list()
    test_list = list()
    for pname in project_list:
        file_name = csv_root + "groundtruth_" + level + "_level_" + pname + ".csv"
        with open(file_name, "r") as f:
            rows = f.read()
            for row in list(set(rows.split("\n")))[1:]:
                if not "," in row:
                    continue
                if "__" in row:
                    continue
                p.append(pname + "#" + row)
                test_list.append(row.split(",")[0].strip())
                d[row.split(",")[0].strip()] = d[row.split(",")[0].strip()] + 1
    a = sorted(d.items(), key=lambda x: x[1])

    test_list = list(set(test_list))
    choose_test = [i[0] for i in a[:int(oo//4)]]
    # print(len(p))
    choose_test.extend(random.sample(p, k=int(oo*3//4)))
    print(len(choose_test))
    for pname in project_list:
        file_name = gt_root + "groundtruth_" + level + "_level_" + pname + ".csv"
        with open(file_name, "w") as f:
            f.write("test,production\n")
            for i in p:
                pp1 = i.split("#")[0]
                pp2 = i.split("#")[1].split(",")[0].strip()
                if pname == pp1 and (pp2 in choose_test or i in choose_test):
                    f.write(i.split("#")[1])
                    f.write("\n")   
                
    d = defaultdict(int)
    for i in p:
        pp1 = i.split("#")[0]
        pp2 = i.split("#")[1].split(",")[0].strip()
        if (pp2 in choose_test or i in choose_test):
            d[pp2] = d[pp2] + 1
    l = sorted(d.values())
    print(sum(l), len(l))
    print(l)
    print(np.mean(l), np.median(l))

if __name__ == "__main__":
    project_list = ["boltons", "face_recognition", "flask", "httpie", "requests", "scrapy", "textdistance"]
    cal1(project_list, "method")
