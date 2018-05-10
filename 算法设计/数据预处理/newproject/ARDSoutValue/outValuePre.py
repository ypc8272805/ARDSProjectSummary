# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

engine= create_engine('postgresql://postgres@localhost:5432/mimic',echo = True)
sql='select * from mimiciii.secondData'
data=pd.read_sql_query(sql, con=engine, index_col=None, coerce_float=True, params=None, parse_dates=None,chunksize=None)
