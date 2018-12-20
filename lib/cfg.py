import os


def get_root_path():
    p = os.path.abspath(__file__)
    i = p.find('/lib/cfg.py')
    assert(i>2)
    return p[:i]
    
root_path = get_root_path()