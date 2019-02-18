# -*- encoding=utf-8 -*-
import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

data = sys.argv[1]#'beam_search_nlpcc_hred_mini_forsearch.txt'
tokenized_sentences = []
outlines =[]
with open(data, 'rb') as f:
    lines = f.readlines()
    outline = ''
    for line in lines:
        line = line.strip().lstrip()
        
        #line = line.decode('utf8')
        line = line.split('\t')[0]+'\n'
        outlines.append(line)
print(len(outlines))
f = open(sys.argv[1]+'___','w')
f.writelines(outlines)
f.close()
