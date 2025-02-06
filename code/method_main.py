from collections import defaultdict
import pytest, csv, Levenshtein, math
from graph.treegraph import TreeGraph as TG
from calltracer import CallTracer
from calltracer.config import Config
from calltracer.globbing_filter import GlobbingFilter
from calltracer.output.gephi import GephiOutput
from calltracer.output.graphviz import GraphvizOutput
import unittest, os, sys, time
import util


def run(project_name, project_root):
    test_files = util.get_test_files(project_root)
    production_files = util.get_production_files(project_root)
    function_in_test, test_function_sum = util.get_function_in_files(test_files, True)
    function_in_production, production_function_sum = util.get_function_in_files(production_files, False)
    code_in_function = util.get_code_in_functions(test_files, function_in_test, production_files, function_in_production)
    git_pair = util.get_git_info_method(project_root.replace("projects", "projects_git"))
    # with open("log.txt", "w") as f:
    #     for pair in git_pair:
    #         f.write(str(pair))
    #         f.write("\n")
    print("test function sum:", test_function_sum, "\nproduction function sum:", production_function_sum)
    technique_list = util.get_technique_list()
    score = dict()
    for tech in technique_list:
        score[tech] = dict()
        for test_file in test_files:
            for tfunc in function_in_test[test_file]:
                test_module = test_file + "@" + tfunc
                score[tech][test_module] = defaultdict(float)
                for production_file in production_files:
                    for pfunc in function_in_production[production_file]:
                        production_module = production_file + "@" + pfunc
                        score[tech][test_module][production_module] = 0.0
    link_production_list = defaultdict(set)
    for test_file in test_files:
        test_file_name = test_file.split("\\")[-1].split(".")[0]
        _tfile = test_file_name.replace("test_", "")
        for tfunc in function_in_test[test_file]:
            test_module = test_file + "@" + tfunc
            print(test_file, tfunc)
            util.call_path = "call_info\\"+project_name+"\\method-level\\"+"@".join(test_file.split("\\")[2:])+ "@" + tfunc+"_call.csv"

            config = Config()
            config.trace_filter = GlobbingFilter(include=[project_name + ".*", test_file_name + ".*", "test.*", "tests.*"])
            call_tracer = CallTracer(config, project_name + "/tests/" + test_file_name, test_file_name, "")
            # run test and trace
            with call_tracer:
                pytest.main(["-q", test_file, "-k", tfunc])

            # suite=unittest.TestSuite()
            # loader = unittest.TestLoader()
            # loader.testMethodPrefix = tfunc
            # discover = loader.discover(project_root, pattern=test_file_name+".py")
            # suite.addTest(discover)
            # with call_tracer:
            #     unittest.TextTestRunner().run(suite)
            
            deep = defaultdict(int)
            # get trace infomation
            with open(util.call_path, "r") as f:
                f_csv = csv.reader(f)
                headers = next(f_csv)
                oo = -1
                for row in f_csv:
                    if "<" in row[0] or ">" in row[0] or "__main__" in row[0]:
                        continue
                    link_production_list[test_module].add(row[0])
                    if row[0].endswith("." + tfunc) and oo == -1:
                        oo = int(row[1])
                        print("--------------------------------------", oo, "--------------------------------------")
                    if oo != -1:
                        deep[row[0]] = int(row[1]) - oo
                    else:
                        deep[row[0]] = 0
                    if test_file_name in row[0] and row[0].endswith(tfunc):
                        lcbas = row[2]
                        if len(lcbas) > 2:
                            lcba_list = lcbas.replace("'", "").replace("[", "").replace("]", "").replace(" ", "").split(";")
                            for lcba in lcba_list:
                                for production_file in production_files:
                                    production_file_name = production_file.split("\\")[-1].split(".")[0]
                                    for pfunc in function_in_production[production_file]:
                                        if production_file_name in lcba.split(".") and lcba.endswith(pfunc):
                                            production_module = production_file + "@" + pfunc
                                            score["LCBA"][test_module][production_module] = 1

            for production_file in production_files:
                production_file_name = production_file.split("\\")[-1].split(".")[0]
                for pfunc in function_in_production[production_file]:
                    cc = 0
                    for key in deep.keys():
                        if production_file_name in key.split(".") and key.endswith("." + pfunc):
                            cc = pow(0.9, deep[key]-1)
                    production_module = production_file + "@" + pfunc
                    _tfunc = tfunc.replace("test_", "")
                    score["NC"][test_module][production_module] = (pfunc == _tfunc) * (1 if cc != 0 else 0)
                    score["NCC"][test_module][production_module] = (pfunc in _tfunc) * (1 if cc != 0 else 0)
                    _lcs = util.lcs(pfunc, _tfunc)
                    score["LCS-B"][test_module][production_module] = (_lcs / max(len(pfunc), len(_tfunc))) * cc
                    score["LCS-U"][test_module][production_module] = (_lcs / len(pfunc)) * cc
                    score["Leven"][test_module][production_module] = (1 - Levenshtein.distance(pfunc, _tfunc) / max(len(pfunc), len(_tfunc))) * cc
                    score["Static NC"][test_module][production_module] = pfunc == _tfunc and _tfile == production_file_name
                    score["Static NCC"][test_module][production_module] = pfunc in _tfunc and production_file_name in _tfile
                    test_fqn = str(test_file.replace(":", "").replace("\\", ".")[:-3] + "." + tfunc).replace("test_", "")
                    production_fqn = production_file.replace(":", "").replace("\\", ".")[:-3] + "." + pfunc
                    _lcs = util.lcs(test_fqn, production_fqn)
                    score["Static LCS-U"][test_module][production_module] = (_lcs / len(production_fqn))
                    score["Static LCS-B"][test_module][production_module] = (_lcs / max(len(production_fqn), len(test_fqn)))
                    score["Static Leven"][test_module][production_module] = (1 - Levenshtein.distance(production_fqn, test_fqn) / max(len(production_fqn), len(test_fqn)))
                    score["Similarity"][test_module][production_module] = (util.calculate_similarity(code_in_function[test_file][tfunc], code_in_function[production_file][pfunc]))
                    score["Co-ev"][test_module][production_module] = (util.cal_appearance_in_git("@".join([test_file_name, tfunc]), "@".join([production_file_name, pfunc]), git_pair))
    print("run over")

    # TFIDF
    print("TFIDF....................")
    for test_file in test_files:
        print("TFIDF " + test_file)
        for tfunc in function_in_test[test_file]:
            test_module = test_file + "@" + tfunc
            for production_file in production_files:
                _production_file = production_file.split("\\")[-1].split(".")[0]
                for pfunc in function_in_production[production_file]:
                    production_module = production_file + "@" + pfunc
                    cnt1 = 0
                    cnt2 = 0
                    for p_file in production_files:
                        __production_file = p_file.split("\\")[-1].split(".")[0]
                        for ppfunc in function_in_production[p_file]:
                            pm = p_file + "@" + ppfunc
                            for item in link_production_list[test_module]:
                                if __production_file in item.split(".") and item.endswith("." + ppfunc):
                                    cnt1 += 1
                    for t_file in test_files:
                        __test_file = t_file.split("\\")[-1].split(".")[0]
                        for ttfunc in function_in_test[t_file]:
                            tm = t_file + "@" + ttfunc
                            for item in link_production_list[tm]:
                               if _production_file in item.split(".") and item.endswith("." + pfunc):
                                    cnt2 += 1
                    aaa = 1
                    if cnt1 == 0 or cnt2 == 0:
                        score["TFIDF"][test_module][production_module] = 0
                        continue
                    aaa = aaa + 1 / cnt1
                    bbb = 1
                    bbb = bbb + production_function_sum / cnt2
                    score["TFIDF"][test_module][production_module] = math.log(aaa) * math.log(bbb)

    # Tarantula
    print("Tarantula....................")
    for test_file in test_files:
        _test_file = test_file.split("\\")[-1].split(".")[0]
        for tfunc in function_in_test[test_file]:
            test_module = test_file + "@" + tfunc
            for production_file in production_files:
                _production_file = production_file.split("\\")[-1].split(".")[0]
                for pfunc in function_in_production[production_file]:
                    production_module = production_file + "@" + pfunc
                    flag = 0
                    for item in link_production_list[test_module]:
                        if _production_file in item.split(".") and item.endswith("." + pfunc):
                            flag = 1
                            break
                    if  flag == 0:
                        score["Tarantula"][test_module][production_module] = 0
                        continue
                    cnt = 0
                    for t_file in test_files:
                        __test_file = t_file.split("\\")[-1].split(".")[0]
                        for ttfunc in function_in_test[t_file]:
                            tm = t_file + "@" + ttfunc
                            for item in link_production_list[tm]:
                               if _production_file in item.split(".") and item.endswith("." + pfunc):
                                    cnt += 1
                    if cnt > 0:
                        cnt = cnt -1
                    score["Tarantula"][test_module][production_module] = 1 / ((cnt / (test_function_sum - 1)) + 1)

    for test_file in test_files:
        for tfunc in function_in_test[test_file]:
            test_module = test_file + "@" + tfunc
            max_score = 0.0
            for production_file in production_files:
                for pfunc in function_in_production[production_file]:
                    production_module = production_file + "@" + pfunc
                    if score["Similarity"][test_module][production_module] > 1.0:
                        print("?!", test_module, production_module, score["Similarity"][test_module][production_module])

    # normalization
    print("normalization.............")
    normalised_list = ["LCS-U", "LCS-B", "Leven", "Tarantula", "TFIDF", "Static LCS-U", "Static LCS-B", "Static Leven", "Similarity", "Co-ev"]
    for j in technique_list:
        print("tech:", j)
        for test_file in test_files:
            for tfunc in function_in_test[test_file]:
                test_module = test_file + "@" + tfunc
                max_score = 0.0
                for production_file in production_files:
                    for pfunc in function_in_production[production_file]:
                        production_module = production_file + "@" + pfunc
                        if max_score < float(score[j][test_module][production_module]):
                            max_score = float(score[j][test_module][production_module])
                if max_score == 0:
                    continue
                for production_file in production_files:
                    for pfunc in function_in_production[production_file]:
                        production_module = production_file + "@" + pfunc
                        aft = float(score[j][test_module][production_module]) / max_score
                        # if score[j][test_module][production_module] > max_score:
                        #     print("???????", j, test_module, production_module, max_score, score[j][test_module][production_module], aft)
                        score[j][test_module][production_module] = aft
    print("write result................")
    pd = defaultdict(bool)
    with open("result/score_method_level_" + project_name + ".csv", "w") as f:
        f.write("test_root,test_method,production_root,production_method")
        for i in technique_list:
            f.write("," + i)
        f.write("\n")
        for test_file in test_files:
            for tfunc in function_in_test[test_file]:
                test_module = test_file + "@" + tfunc
                for production_file in production_files:
                    for pfunc in function_in_production[production_file]:
                        production_module = production_file + "@" + pfunc
                        if pd[test_module + "#" + production_module]:
                            continue
                        pd[test_module + "#" + production_module] = True
                        f.write(test_module.replace("@", ",") + "," + production_module.replace("@", ","))
                        for j in technique_list:
                            f.write("," + str(float(score[j][test_module][production_module])))
                        f.write("\n")


def test_run(project_name, project_root):
    test_files = util.get_test_files(project_root)
    production_files = util.get_production_files(project_root)
    function_in_test, test_function_sum = util.get_function_in_files(test_files, True)
    function_in_production, production_function_sum = util.get_function_in_files(production_files, False)
    code_in_function = util.get_words_in_files(test_files, function_in_test, production_files, function_in_production)
    print(test_files[0])
    for item in code_in_function[test_files[0]]:
        print("!!!", item, ":")
        print(code_in_function[test_files[0]][item])
    print(util.calculate_similarity(code_in_function[test_files[0]]["test_basic_url_generation"], code_in_function[test_files[0]]["test_url_generation_requires_server_name"]))


if __name__ == "__main__":
    startTime = time.time()
    project_name = "boltons"
    project_root = "projects_new\\" + project_name
    run(project_name, project_root)
    # util.get_git_info_method(project_root)
    # test_files = util.get_test_files(project_root)
    # production_files = util.get_production_files(project_root)
    # function_in_test, tfunction_sum = util.get_function_in_files(test_files)
    # function_in_production, production_function_sum = util.get_function_in_files(production_files)
    # print("test function sum:", tfunction_sum, "\nproduction function sum:", production_function_sum)
    endTime = time.time()
    print("start time:", time.asctime(time.localtime(startTime)))
    print("end time:", time.asctime(time.localtime(endTime)))
    print("use time:", (int)(endTime - startTime), "s\n")
    