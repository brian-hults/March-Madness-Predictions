# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:56:31 2021

NCAA Mens Basketball - SQL database utilities

@author: Brian Hults
"""

# Install pacakges
# import pandas as pd
import sqlite3 as sql
from sqlite3 import Error

class SQL_Utils:
    def __init__(self):
        self.db_file = './sqlite/db/team_schedule_data.db'
        self.conn = self.create_connection()
        
    def create_connection(self):
        conn = None
        try:
            conn = sql.connect(self.db_file)
        except Error as e:
            print(e)
        return conn
    
    def create_table(self, create_table_sql):
        try:
            c = self.conn.cursor()
            c.execute(create_table_sql)
        except Error as e:
            print(e)