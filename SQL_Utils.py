# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:56:31 2021

NCAA Mens Basketball - SQL database utilities

@author: Brian Hults
"""

# Install pacakges
import sqlalchemy as sql
import sqlalchemy_utils.functions as sql_utils

class SQL_Utils:
    def __init__(self):
        self.db_address = 'sqlite:///db/NCAA_Basketball.db'
        if not sql_utils.database_exists(self.db_address):
            sql_utils.create_database(self.db_address)
        self.engine = self.create_engine()
        
    def create_engine(self):
        engine = sql.create_engine(self.db_address, echo=False, pool_recycle=1800)
        return engine
    