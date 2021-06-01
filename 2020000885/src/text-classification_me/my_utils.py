# -*- encoding: utf-8 -*-

import json
f=open("filename.json","r")
data=list()
for line in f:
    data.append(json.loads(line))
f.close()