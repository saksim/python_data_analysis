#!-*-encoding = utf-8-*-
import sys
reload(sys)
sys.getdefaultencoding()
sys.setdefaultencoding('utf-8')

from urllib import urlopen
from bs4 import BeautifulSoup
import re,string
from collections import OrderedDict

def cleaninput(input):
    input = re.sub('\n+'," ",input)
    input = re.sub(' +'," ",input)
    input = re.sub('\[[0-9]*\]',"",input)
    input = bytes(input.encode('utf-8'))
    input = input.decode("ascii","ignore")
    cleaninput = []
    input = input.split(' ')
    for i in input:
        i = i.strip(string.punctuation)
        if len(i) > 1 or (i.lower() == 'a' or i.lower() == 'i'):
            cleaninput.append(i)
    return cleaninput

def ngrams(input, n):
    input = cleaninput(input)
    output = []
    for i in range(len(input)-n+1):
        output.append(input[i:i+n])
    return output

if __name__ == "__main__":
    html = urlopen("http://en.wikipedia.org/wiki/Python_(programming_language)")
    bsobj = BeautifulSoup(html,"html5lib")
    content = bsobj.find("div",{"id":"mw-content-text"}).get_text()
    ngram = ngrams(content, 2)
    ngram = dict(ngram)
    ngram = OrderedDict(sorted(ngram.items(),key=lambda t:t[1],reverse=True))
    print ngram
    print("2-grams count is :" + str(len(ngram)))
