
import sys
import math

def totalP():
    unigram_count = {}
    token_num = 0.0
    with open('/data/wangyanmeng/icassp/data/raw_train_wt_command') as fin:
        lines_train = fin.read().decode('utf8').split('\n')
        for line in lines_train:
            token = line.strip().split()
            token_num += len(token)
            for word in token:
                if word in unigram_count:
                    temp = unigram_count[word] + 1.0
                    unigram_count[word] = temp
                else:
                    unigram_count[word] = 1.0
    return unigram_count, token_num
def count_diversity(lines):
    
    unigram = set()
    bigram = set()
    token_num = 0.0
    unigram_count = {}
    entropy = 0.0
    train_total_num = 0.0
    unigram_count, train_total_num = totalP()
    
    for line in lines:
        #print line
        #print type(line)
        token = line.strip().split()
        
        token_num += len(token)
        unigram.update(token)
        bi = [''.join(token[i:i+2]) for i in range(len(token)-1)]
        bigram.update(bi)
    ue = 0.0
    for word in unigram:
        #token = line.strip().split()
        
        
        #for word in token:
        if word in unigram_count:    
            p = unigram_count[word]/train_total_num
            #print p
            ue = ue- p * math.log(p,2)
            #print p * math.log(p,2)
    entropy = ue
    return len(unigram), len(bigram), token_num, entropy

def linesToLow(lines):
    outlines = []
    for line in lines:
        outlines.append(line.lower())
    return outlines    
if __name__ == '__main__':
    f = sys.argv[1]
    with open(f) as fin:
        lines = fin.read().decode('utf8').split('\n')
    lines1 = linesToLow(lines)
    un_num, bi_num, total_num, entropy = count_diversity(lines1)
    print 'unique unigram : %s'%un_num
    print 'unique bigram  : %s'%bi_num
    print 'total token    : %s'%total_num
    print 'distinct-1: %s'%(1.0*un_num / total_num)
    print 'distinct-2: %s'%(1.0*bi_num / total_num)
    print 'entropy H_w: %s'%(entropy)
    print 'entropy H_U: %s'%(entropy*total_num/11594)    
