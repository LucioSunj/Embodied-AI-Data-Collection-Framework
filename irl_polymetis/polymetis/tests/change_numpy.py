import numpy as np
file_path = '/home/lsk/irl_polymetis/D435i_extrinsics_chess_right.npy'
data = np.load(file_path)
data[1,3] += 0.025
np.save('D435i_extrinsics_chess_right', data)
print(data)