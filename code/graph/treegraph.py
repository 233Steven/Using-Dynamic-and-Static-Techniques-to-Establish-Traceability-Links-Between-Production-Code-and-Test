import os
# import UnitTestTracing.config as cf

from typing import List
from types import FunctionType
from .treenode import TreeNode as TN
from queue import Queue

class TreeGraph:
    def __init__(self, dir: str, onlyPythonFile: bool = True) -> None:
        if dir[-1] == cf.FILE_SLASH:
            self.dir = dir[:-1]
        else:
            self.dir = dir
        self.projectName = self.dir.split("\\")[-1]
        self.testNodes = list()
        self.productionNodes = list()
        self.links = list()
        self.path_to_node = dict()
        self.solve_graph(onlyPythonFile)

    def add_node(self, _absolutePath: str, fatherNode: TN = None) -> TN:
        newNode = TN(_absolutePath, fatherNode)
        return newNode
    
    def get_dir(self) -> str:
        return self.dir
        
    def get_project_name(self) -> str:
        return self.projectName

    def get_test_nodes(self) -> List[TN]:
        return self.testNodes
    
    def get_test_files(self) -> List[str]:
        result = list()
        for node in self.get_test_nodes():
            result.append(node.get_absolute_path())
        return result

    def get_production_nodes(self) -> List[TN]:
        return self.productionNodes

    def get_production_files(self) -> List[str]:
        result = list()
        for node in self.get_production_nodes():
            result.append(node.get_absolute_path())
        return result
    
    def get_python_nodes(self) -> List[TN]:
        result = list()
        result.extend(self.get_production_nodes())
        result.extend(self.get_test_nodes())
        return result
    
    def get_python_files(self) -> List[str]:
        result = list()
        for node in self.get_python_nodes():
            result.append(node.get_absolute_path())
        return result

    def get_node_by_path(self, path: str) -> TN:
        return self.path_to_node.get(path, None)
        # allNodes = self.get_python_nodes()
        # for node in allNodes:
        #     print(node.get_absolute_path())
        #     if node.get_absolute_path() == path:
        #         return node
        # return None
    
    def get_nodes_by_path_suffix(self, path_suffix: str) -> List[TN]:
        result = list()
        nodes = self.get_python_nodes()
        for node in nodes:
            path = node.get_absolute_path()
            if path.endswith(path_suffix):
                result.append(node)
        return result
            
    def solve_graph(self, onlyPythonFile: bool) -> None:
        self.root = self.add_node(self.dir)
        que = Queue()
        que.put(self.root)
        while que.empty() == False:
            nowNode = que.get()
            nowPath = nowNode.get_absolute_path()
            self.path_to_node[nowPath] = nowNode
            files = os.listdir(nowPath)
            for f in files:
                f_d = os.path.join(nowPath, f)
                if onlyPythonFile and os.path.isfile(f_d) and not f.endswith('.py'):
                    continue
                nxtNode = self.add_node(f_d, nowNode)
                self.path_to_node[nxtNode.get_absolute_path()] = nxtNode
                if nxtNode.is_test():
                    self.testNodes.append(nxtNode)
                if nxtNode.is_production():
                    self.productionNodes.append(nxtNode)
                if os.path.isdir(f_d):
                    que.put(nxtNode)
    
    def dfs_tree_node(self, nowNode: TN, func: FunctionType) -> None:
        func(nowNode)
        childNode = nowNode.firstChild
        while childNode:
            self.dfs_tree_node(childNode, func)
            childNode = childNode.nextBrother
    
    def bfs_tree_node(self, rootNode: TN, func: FunctionType) -> None:
        que = Queue()
        que.put(rootNode)
        while que.empty() == False:
            nowNode = que.get()
            func(nowNode)
            nxtNode = nowNode.firstChild
            while nxtNode:
                que.put(nxtNode)
                nxtNode = nxtNode.nextBrother

    def output_tree(self, outputType: str) -> bool:     # Two way to output tree: dfs or bfs
        if outputType not in ['dfs', 'bfs']:            # return: if output or not
            return False
        if outputType == 'dfs':
            self.dfs_tree_node(self.root, print)
            return True
        elif outputType == 'bfs':
            self.bfs_tree_node(self.root, print)
            return True

# if __name__ == '__main__':
#     tg = TreeGraph(cf.PROJECTS_DIR[0])
    # tg.output_tree('dfs')