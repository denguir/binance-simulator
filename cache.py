import os
import json
import pathlib
import inspect
import pandas as pd
import joblib

# when get_symbol_data is called:
# 1- list all the called get_symbol_data with same (symbol, resolution)
# 2- compute intersection with called date intervals and query date interval
# 3- call cached fct for called date intervals
# 4- call api fct for the rest 

# cache decorator for load_symbol_data
# that will break the call into sub calls


class CacheManager():

    def __init__(self, cachedir, verbose=0):
        self.cachedir = cachedir
        self.verbose = verbose
        self.memory = joblib.Memory(cachedir, verbose=verbose)

    def get_cache_dir(self, func):
        return pathlib.Path(self.cachedir) \
            / joblib.__name__ / func.__module__ / func.__name__

    def get_func_args(self, func, ignore=[]):
        return list(filter(lambda x: x not in ignore, inspect.getfullargspec(func).args))

    def get_cache_metadata(self, func):
        df_meta = pd.DataFrame(
            columns=self.get_func_args(func, ignore=['self']))
        cachedir = self.get_cache_dir(func)
        try:
            for dir in os.listdir(cachedir):
                if os.path.isdir(cachedir / dir):
                    with open(cachedir / dir / "metadata.json") as meta_file:    
                        meta_data = json.load(meta_file)
                        meta_args = {k:v.strip("''") for k,v in meta_data['input_args'].items()}
                        df_meta = df_meta.append(meta_args, ignore_index=True)
        except Exception as e:
            print('Cache retrieve went wrong.')
            print(e)         
        return df_meta

    def cache(self, func=None, ignore=[]):
        if type(func) is joblib.memory.MemorizedFunc:
            func = func.func
        
        match_args = self.get_func_args(func, ignore=ignore + ['date_from', 'date_to'])
        df_meta = self.get_cache_metadata(func)

        def get_similar_calls(**kwargs):
            filter_meta = {k:v for k,v in kwargs.items() if k in match_args}
            df = df_meta.loc[(df_meta[list(filter_meta)] == pd.Series(filter_meta)).all(axis=1)]
            return df

        def get_intersection_calls(df):
            # this one will be a pain is the ass
            pass

        def wrapper(**kwargs):
            df_filtered = get_similar_calls(kwargs)
            cached_calls = get_intersection_calls(df_filtered)
            
            





