#!/usr/bin/env python


#Python standard
import os
import pickle

def pickle_cache(func):

    def pfunc(*args, **kwargs):

        if 'path' in kwargs:
            path = kwargs['path']
            del kwargs['path']

            if os.path.exists(path):
                with open(path, 'rb') as hf:
                    _res = pickle.load(hf)
            else:

                _res = func(*args, **kwargs)

                with open(path, 'wb') as hf:
                    pickle.dump(_res, hf)

            return _res
        else:
            return func(*args, **kwargs)

    return pfunc
                    
