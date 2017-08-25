#!-*-encoding=utf-8-*-
from urllib2 import urlopen
from bs4 import BeautifulSoup
import pymysql

conn = pymysql.connect(host='127.0.0.1',user='root',passwd='MYFyxy5hww21',database='python_test',port = 28080,charset='utf8')
cur = conn.cursor()
cur.execute('USE wikioedia')

class SolutionFound(RuntimeError):
    def __init__(self,message):
        self.message = message

def getLinks(fromPageid):
    cur.execute("")
