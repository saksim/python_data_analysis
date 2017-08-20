#!-*-encoding = utf-8-*-
import sys
reload(sys)
sys.getdefaultencoding()
sys.setdefaultencoding('utf-8')

from urllib import urlopen
from bs4 import BeautifulSoup
import re,string
from collections import OrderedDict
import operator

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
    output = {}
    for i in range(len(input)-n+1):
        ngramtemp = "".join(input[i:i+n])
        if ngramtemp not in output:
            output[ngramtemp] = 0
        output[ngramtemp] += 1
    return output

#有意义词汇与无意义词汇的差异，添加isConnon函数实现筛选

def isCommon(ngram):
    commonWords = ["the", "be", "and", "of", "a", "in", "to", "have",
                   "it","i", "that", "for", "you", "he", "with", "on",
                   "do", "say", "this","they", "is", "an", "at", "but",
                   "we", "his", "from", "that", "not","by", "she", "or",
                   "as", "what", "go", "their","can", "who", "get","if",
                   "would", "her", "all", "my", "make", "about", "know",
                   "will","as", "up", "one", "time", "has", "been", "there",
                   "year", "so","think", "when", "which", "them", "some", "me",
                   "people", "take","out", "into", "just", "see", "him", "your",
                   "come", "could", "now","than", "like", "other", "how", "then",
                   "its", "our", "two", "more","these", "want", "way", "look", "first",
                   "also", "new", "because","day", "more", "use", "no",
                   "man", "find", "here", "thing", "give","many", "well"]
    for word in ngram:
        if word in commonWords:
            return True
    return False

if __name__ == "__main__":
    # html = urlopen("http://en.wikipedia.org/wiki/Python_(programming_language)")
    # bsobj = BeautifulSoup(html,"html5lib")
    # content = bsobj.find("div",{"id":"mw-content-text"}).get_text()
    content = str(urlopen("http://pythonscraping.com/files/inaugurationSpeech.txt").read())
    ngram = ngrams(content, 2)
    # ngram = dict(ngram)
    ngram = sorted(ngram.items(),key=operator.itemgetter(1),reverse=True)
    print ngram
    print("2-grams count is :" + str(len(ngram)))

