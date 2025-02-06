from collections import defaultdict
import pytest, re, csv ,Levenshtein, math
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
    code_in_file = util.get_code_in_files(test_files, production_files)
    git_pair = util.get_git_info_file(project_root.replace("projects", "projects_git"))
    print("test function sum:", test_function_sum, "\nproduction function sum:", production_function_sum)
    technique_list = util.get_technique_list()
    score = dict()
    for tech in technique_list:
        score[tech] = dict()
        for test_file in test_files:
            test_file_name = test_file.split("\\")[-1].split(".")[0]
            score[tech][test_file] = defaultdict(float)
    link_production_list = defaultdict(set)
    for test_file in test_files:
        test_file_name = test_file.split("\\")[-1].split(".")[0]
        _tfile = test_file_name.replace("test_", "")
        print(test_file)
        util.call_path = "call_info\\"+project_name+"\\file-level\\" + "@".join(test_file.split("\\")[2:]) + "_call.csv"
        config = Config()
        config.trace_filter = GlobbingFilter(include=[project_name + ".*", test_file_name + ".*", "test.*", "tests.*"])
        call_tracer = CallTracer(config, project_name + "/tests/" + test_file_name, test_file_name, "")
        # run test and trace
        with call_tracer:
            pytest.main(["-q", test_file])

        # suite=unittest.TestSuite()
        # loader = unittest.TestLoader()
        # discover = loader.discover(project_root, pattern = test_file_name + ".py")
        # suite.addTest(discover)
        # with call_tracer:
        #     unittest.TextTestRunner().run(suite)
            
        deep = defaultdict(int)
        with open(util.call_path, "r") as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            for row in f_csv:
                if "<" in row[0] or ">" in row[0] or "__main__" in row[0]:
                    continue
                for production_file in production_files:
                    production_file_name = production_file.split("\\")[-1].split(".")[0]
                    if production_file_name in row[0].split("."):
                        link_production_list[test_file_name].add(production_file_name)
                deep[row[0]] = int(row[1]) - 31
                if test_file_name in row[0].split("."):
                    lcbas = row[2]
                    # with open("log.txt", "a") as f:
                    #     f.write(_test_file + ", " + tfunc + ", " + lcbas + "\n")
                    if len(lcbas) > 2:
                        lcba_list = lcbas.replace("'", "").replace("[", "").replace("]", "").replace(" ", "").split(";")
                        for lcba in lcba_list:
                            if lcba.endswith("__"):
                                continue
                            for production_file in production_files:
                                production_file_name = production_file.split("\\")[-1].split(".")[0]
                                if production_file_name in lcba:
                                    score["LCBA"][test_file][production_file] = 1
                                    # with open("log.txt", "a") as f:
                                    #     f.write(_test_file + ", " + tfunc + ", " + _production_file + ", " + pfunc + ", " + lcba + "\n")
        for production_file in production_files:
            production_file_name = production_file.split("\\")[-1].split(".")[0]
            cc = 0
            for key in deep.keys():
                if production_file_name in key.split("."):
                    # with open("log.txt", "a") as f:
                    #     f.write(_production_file + " " + pfunc + str(deep[key]) + "\n")
                    cc = pow(0.9, deep[key]-1)
            score["NC"][test_file][production_file] = (production_file_name == _tfile) * (1 if cc != 0 else 0)
            score["NCC"][test_file][production_file] = (production_file_name in _tfile) * (1 if cc != 0 else 0)
            _lcs = util.lcs(production_file_name, _tfile)
            score["LCS-B"][test_file][production_file] = (_lcs / max(len(production_file_name), len(_tfile))) * cc
            score["LCS-U"][test_file][production_file] = (_lcs / len(production_file_name)) * cc
            score["Leven"][test_file][production_file] = (1 - Levenshtein.distance(production_file_name, _tfile) / max(len(production_file_name), len(_tfile))) * cc
            score["Static NC"][test_file][production_file] = production_file_name == _tfile
            score["Static NCC"][test_file][production_file] = production_file_name in _tfile
            test_fqn = str(test_file.replace(":", "").replace("\\", ".")[:-3]).replace("test_", "")
            production_fqn = production_file.replace(":", "").replace("\\", ".")[:-3]
            _lcs = util.lcs(test_fqn, production_fqn)
            score["Static LCS-U"][test_file][production_file] = _lcs / len(production_fqn)
            score["Static LCS-B"][test_file][production_file] = _lcs / max(len(production_fqn), len(test_fqn))
            score["Static Leven"][test_file][production_file] = 1 - Levenshtein.distance(production_fqn, test_fqn) / max(len(production_fqn), len(test_fqn))
            score["Similarity"][test_file][production_file] = util.calculate_similarity(code_in_file[test_file], code_in_file[production_file])
            score["Co-ev"][test_file][production_file] = util.cal_appearance_in_git(test_file_name, production_file_name, git_pair)

    # TFIDF
    print("TFIDF....................")
    for test_file in test_files:
        _test_file = test_file.split("\\")[-1].split(".")[0]
        # print("TFIDF " + _test_file)
        for production_file in production_files:
            _production_file = production_file.split("\\")[-1].split(".")[0]
            cnt1 = 0
            cnt2 = 0
            for p_file in production_files:
                __production_file = p_file.split("\\")[-1].split(".")[0]
                for item in link_production_list[_test_file]:
                    if __production_file in item.split("."):
                        cnt1 += 1
            for t_file in test_files:
                __test_file = t_file.split("\\")[-1].split(".")[0]
                for item in link_production_list[__test_file]:
                    if _production_file in item.split("."):
                        cnt2 += 1
            aaa = 1
            if cnt1 == 0 or cnt2 == 0:
                score["TFIDF"][test_file][production_file] = 0
                continue
            aaa = aaa + 1 / cnt1
            bbb = 1
            bbb = bbb + production_function_sum / cnt2
            score["TFIDF"][test_file][production_file] = math.log(aaa) * math.log(bbb)

    # Tarantula
    print("Tarantula....................")
    for test_file in test_files:
        _test_file = test_file.split("\\")[-1].split(".")[0]
        for production_file in production_files:
            _production_file = production_file.split("\\")[-1].split(".")[0]
            flag = 0
            for item in link_production_list[_test_file]:
                if _production_file in item.split("."):
                    flag = 1
                    break
            if flag == 0:
                score["Tarantula"][test_file][production_file] = 0
                continue
            cnt = 0
            for t_file in test_files:
                __test_file = t_file.split("\\")[-1].split(".")[0]
                for item in link_production_list[__test_file]:
                    if _production_file in item.split("."):
                        cnt += 1
            if cnt > 0:
                cnt = cnt -1
            score["Tarantula"][test_file][production_file] = 1 / ((cnt / (test_function_sum - 1)) + 1)

    # normalization
    print("normalization.............")
    normalised_list = ["LCS-U", "LCS-B", "Leven", "Tarantula", "TFIDF", "Static LCS-U", "Static LCS-B", "Static Leven", "Similarity", "Co-ev"]
    for j in normalised_list:
        print("tech:", j)
        max_score = 0.0
        for test_file in test_files:
            _test_file = test_file.split("\\")[-1].split(".")[0]
            for production_file in production_files:
                _production_file = production_file.split("\\")[-1].split(".")[0]
                max_score = max(max_score, float(score[j][test_file][production_file]))
        print("max_score:", max_score)
        if max_score == 0:
            continue
        for test_file in test_files:
            _test_file = test_file.split("\\")[-1].split(".")[0]
            for production_file in production_files:
                _production_file = production_file.split("\\")[-1].split(".")[0]
                aft = score[j][test_file][production_file] / max_score
                if aft > 1.0:
                    print("???????", j, test_file, production_file, max_score, score[j][_test_file][_production_file], aft)
                score[j][test_file][production_file] = aft

    # write result
    print("write result................")
    pd = defaultdict(bool)
    with open("result/score_file_level_" + project_name + ".csv", "w") as f:
        f.write("test_file,production_file")
        for i in technique_list:
            f.write("," + i)
        f.write("\n")
        for test_file in test_files:
            _test_file = test_file.split("\\")[-1].split(".")[0]
            if _test_file.startswith("__"):
                continue
            for production_file in production_files:
                _production_file = production_file.split("\\")[-1].split(".")[0]
                if _production_file.startswith("__"):
                    continue
                if pd[_test_file + "#" + _production_file]:
                    continue
                pd[_test_file + "#" + _production_file] = True
                f.write(test_file + "," + production_file)
                for j in technique_list:
                    f.write("," + str(float(score[j][test_file][production_file])))
                f.write("\n")


def test_run(project_name, project_root):
    test_files = util.get_test_files(project_root)
    production_files = util.get_production_files(project_root)
    function_in_test, test_function_sum = util.get_function_in_files(test_files, True)
    function_in_production, production_function_sum = util.get_function_in_files(production_files, False)
    print("test function sum:", test_function_sum, "\nproduction function sum:", production_function_sum)
    # sys.path.append(project_root)
    for test_file in test_files:
        test_file_name = test_file.split("\\")[-1].split(".")[0]
        print(test_file)
        config = Config()
        config.trace_filter = GlobbingFilter(include=[project_name+".*","*"+test_file_name+".*","tests.*","test.*"])
        # config.trace_filter = GlobbingFilter()
        with CallTracer(config, test_file_name):
            pytest.main(["-q", test_file])
        
        # suite=unittest.TestSuite()
        # loader = unittest.TestLoader()
        # loader.testMethodPrefix = tfunc
        # discover = loader.discover(project_root, pattern=test_file_name+".py")
        # suite.addTest(discover)
        # with CallTracer(config):
        #     unittest.TextTestRunner().run(suite)



if __name__ == "__main__":
    startTime = time.time()
    project_name = "boltons"
    project_root = "projects_new\\" + project_name
    run(project_name, project_root)
    # test_files = util.get_test_files(project_root)
    # production_files = util.get_production_files(project_root)
    # function_in_test, tfunction_sum = util.get_function_in_files(test_files)
    # function_in_production, production_function_sum = util.get_function_in_files(production_files)
    # print("test function sum:", tfunction_sum, "\nproduction function sum:", production_function_sum)
    endTime = time.time()
    print("start time:", time.asctime(time.localtime(startTime)))
    print("end time:", time.asctime(time.localtime(endTime)))
    print("use time:", (int)(endTime - startTime), "s\n")
    