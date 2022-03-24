
import numpy as np
import torch.nn as nn
import torch
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

bpath = '/home/peter/workspace/projects/BACKUP_DONE/tradance/traditional-dance-recognition/logs/kordance600_13-rgb-i3d-resnet-18-ts-f32'

valpath= '/home/peter/workspace/dataset/kor_dance/kordance600_13/val.txt'

with open(valpath,'r') as f:
    vallist = [ss.rstrip().split() for ss in f.readlines()]

scorepath = bpath+'/val_1crops_1clips_224_details.npy'
labelpath = bpath+'/labels.npy'

score= np.load(scorepath)
label =np.load(labelpath)


m = nn.Softmax(dim=1)
res = m(torch.from_numpy(score))

print(res[0])


aa = res.data.cpu().numpy()

print(aa.shape, label.shape, len(vallist))
count = 0
for i in range(200):
    s =  aa[i]
    l = label[i]
    v = vallist[i]

    if np.argmax(s) == l:
        count +=1
        s.sort()
        s=s[::-1]
        print(v[0] , s[:5])
print(count, ' corrected')
# for a in aa:
    # print(a)