
import numpy as np



a ='/home/peter/workspace/projects/tradance/traditional-dance-recognition/logs/kordance600_13-rgb-i3d-resnet-18-ts-f32/val_3crops_3clips_224_details.npy'

npy = np.load(a)

print(npy.shape)
print(npy[0])

