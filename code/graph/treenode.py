import os
# import UnitTestTracing.config as cf

class TreeNode:
    def __init__(self, absolutePath: str, father = None, read: bool = True) -> None:
        self.absolutePath = absolutePath
        self.value = ''
        self.name = self.absolutePath.split("\\")[-1]
        self.isDir = os.path.isdir(self.absolutePath)
        self.isFile = os.path.isfile(self.absolutePath)
        self.isPython = False
        self.isTest = False
        self.isProduction = False
        self.nameSuffix = ''
        if self.isFile:
            self.isFile = True
            if '.' in self.name:
                self.nameSuffix = self.name.split('.')[1]
                if self.nameSuffix == 'py' or self.nameSuffix == 'pyx':
                    self.isPython = True
                    if read:
                        with open(self.absolutePath, 'r', encoding='utf-8') as f:
                            self.value = f.read()
                    if self.name.startswith('test_'):   # test file
                        self.isTest = True
                    else:                               # production file
                        self.isProduction = True
        self.father = father       # Node which is selfNode's father
        self.nextBrother = None     # Node which is next to selfNode
        self.firstChild = None      # Node which is selfNode's first child
        self.isRoot = False
        self.deep = 0
        if self.father == None:
            self.isRoot = True
        else:
            self.nextBrother = self.father.firstChild
            self.father.firstChild = self
            self.deep = self.father.deep + 1

    def is_dir(self) -> bool:
        return self.isDir

    def is_file(self) -> bool:
        return self.isFile
    
    def is_python(self) -> bool:
        return self.isPython

    def is_root(self) -> bool:
        return self.isRoot
    
    def is_test(self) -> bool:
        return self.isTest
    
    def is_production(self) -> bool:
        return self.isProduction

    def get_value(self) -> str:
        return self.value

    def get_absolute_path(self) -> str:
        return self.absolutePath

    def get_name(self) -> str:
        return self.name

    def get_name_no_suffix(self) -> str:
        return self.name.split('.')[0]

    def get_package(self) -> str:
        ans = self.name.split('.')[0]
        it = self
        while it.isRoot == False:
            it = it.father
            ans = it.name + '.' + ans
        return ans
    
    def link_node(self, node) -> None:
        self.linkNodes.append(node)

    def __hash__(self) -> int:
        return hash(self.get_absolute_path())

    def __eq__(self, o: object) -> bool:
        if isinstance(o, TreeNode) and self.absolutePath == o.absolutePath:
            return True
        else:
            return False

    def __repr__(self) -> str:
        # out = '-' * self.deep
        # out += self.name
        # return out
        return self.get_absolute_path()