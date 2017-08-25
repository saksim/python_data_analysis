#!-*-encoding=utf-8-*-
from urllib import urlopen
from bs4 import BeautifulSoup
import re
import datetime
import random
import pymysql

conn = pymysql.connect(host='127.0.0.1', user='root', passwd='MYFyxy5hww21', database='python_test', port=28080,
                       charset='utf8')
cur = conn.cursor()

def crete_table():
    statement = "CREATE TABLE pages (id BIGINT(7) NOT NULL AUTO_INCREMENT,title VARCHAR(200),content VARCHAR(10000),created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,PRIMARY KEY(id));"
    # print statement
    cur.execute(statement)
    cur.connection.commit()

def store(title, content):
    cur.execute("INSERT INTO pages (title, content) VALUES (\"%s\",\"%s\")", (title,content))
    cur.connection.commit()

def getLinks(articleurl):
    html = urlopen("http://en.wikipedia.org/wiki" + articleurl)
    bsobj = BeautifulSoup(html,"html5lib")
    title = bsobj.find("h1").get_text()
    for i in bsobj.find("div",{"id":"mw-content-text"}).find("p"):
        content = i.get_text()
        store(title,content)
    return bsobj.find("div",{"id":"bodyContent"}).findall("a",href =re.compile("^(/wiki/)"))

links = getLinks("/Climate")

if __name__ == "__main__":
    random.seed(datetime.datetime.now())
    # crete_table()
    try:
        while len(links) > 0:
            newarticle = links[random.randint(0,len(links)-1)].attrs["href"]
            print(newarticle)
            links = getLinks(newarticle)
    finally:
        cur.close()
        conn.close()