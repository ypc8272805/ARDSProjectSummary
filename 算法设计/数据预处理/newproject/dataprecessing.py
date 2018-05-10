# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:09:08 2018

@author: zg
"""

from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt

engine= create_engine('postgresql://postgres@localhost:5432/mimic',echo = True)
sql='select * from mimiciii.secondData'
df=pd.read_sql_query(sql, con=engine, index_col=None, coerce_float=True, params=None, parse_dates=None,chunksize=None)
nbps=df['nbps']