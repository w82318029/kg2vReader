import sys

f = open('/data/wangyanmeng/icassp/data/entityName980')
lines = f.readlines()
f.close()
entityName = set()
for line in lines:
	line = line.strip()
	entityName.add(line)
count = 0
f = open(sys.argv[1])
lines = f.readlines()
f.close()
for line in lines:
	line = line.strip()
	words = line.split()
	for word in words:
		if word in entityName:
			count+=1
print "entityname number is %d"%count