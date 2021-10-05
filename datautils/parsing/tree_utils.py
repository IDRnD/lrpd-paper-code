from typing import List
from pathlib import Path
from collections import defaultdict

def build_tree():
    """
    Builds infinite dict or tree structure.
    """
    tree = lambda: defaultdict(tree)
    instance = tree()
    return instance

class Tree(object):
    def __init__(self):
        self.__tree = build_tree()
        
    def __getitem__(self,k):
        return recursive_get(self.__tree,Path(k))
    
    def __setitem__(self,k,v):
        recursive_set(self.__tree,v,Path(k))
        
    def __repr__(self):
        return self.__tree.__repr__()

def __recursive_get__(tree,index):
    recursive_get(tree,Path(index))

def __recursive_set__(tree,index,value):
    recursive_set(tree,value,Path(index))

def recursive_set(ndict,value,path:Path):
    _parts = list(path.parts)[::-1]
    def _rset(d,v):
        if len(_parts) > 1:
            _rset(d[_parts.pop()],v)
        else:
            d[_parts[0]] = v
    _rset(ndict,value)

def recursive_get(ndict,path:Path):
    _parts = list(path.parts)[::-1]
    def _rget(d):
        if len(_parts) > 1:
            k = _parts.pop()
            if k in d:
                return _rget(d[k])
            else:
                raise KeyError()
        else:
            return d[_parts[0]]
    return _rget(ndict)