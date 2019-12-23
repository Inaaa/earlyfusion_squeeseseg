import os
import numpy as np

lidar_2d_path = os.path.join('/home/chli/cc_code/SqueezeSeg', 'data/lidar_2d/2011_09_26_0001_0000000000.npy')

print(lidar_2d_path)

record = np.load(lidar_2d_path).astype(np.float32, copy= False)
print(record[ :,:5])
print(record.shape)

record = record[:, ::-1, :]
print(record.shape)
print(record[:, :5])

record[:, :, 1] *= -1
print(record.shape)
print(record[:, :5])
lidar = record[:, :, :5]
lidar_mask = np.reshape(
    (lidar[:, :, 4] > 0),
    [64, 512, 1]
)
print(lidar_mask)