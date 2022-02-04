# MIT License
# 
# Copyright (c) 2022, Alex M. Maldonado
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import numpy as np
from mbgdml.data import dataSet
from mbgdml.train import mbGDMLTrain

##### Job Information #####

dataset_dir = '../../data/datasets'
dataset_dir_h2o = f'{dataset_dir}/h2o'
dataset_dir_mecn = f'{dataset_dir}/mecn'
dataset_dir_meoh = f'{dataset_dir}/meoh'

model_save_dir = '../../data/models'
model_save_dir_h2o = f'{model_save_dir}/h2o'
model_save_dir_mecn = f'{model_save_dir}/mecn'
model_save_dir_meoh = f'{model_save_dir}/meoh'

dataset_dir_solvent = dataset_dir_meoh
model_save_dir_solvent = model_save_dir_meoh

dataset_path = f'{dataset_dir_solvent}/2meoh/62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh-dset.mb-cm8.npz'
save_dir = f'{model_save_dir_solvent}/2meoh/'

dset = dataSet(dataset_path)
total_number = dset.n_R  # Number to total structures in data set.
num_train = 1000  # Number of data points to train on.
num_validate = 2000  # Number of data points to compare candidate models with.
num_test = total_number - num_train - num_validate  # Number of data points to test final model with.
if num_test > 3000:
    num_test = 3000
sigmas = list(range(470, 602, 5))  # Sigmas to train models on.
use_E_cstr = False  # Adds energy constraints to the kernel. `False` usually provides better performance. `True` can help converge training in rare cases.
overwrite = True  # Whether the script will overwrite a model.
torch = False  # Whether to use torch to accelerate training.
model_name = f'62meoh.sphere.gfn2.md.500k.prod1.3meoh.cm14.dset.2meoh.cm8-model.mb-iterativetrain{num_train}.2'

# None for automatically selected or specify a list.
idxs_train = np.array(
    [ 2174, 8551, 8802, 4940, 4948, 1315, 4998, 6090, 3378, 6562, 890, 5575, 3787, 7482, 2516, 8166, 3065, 1830, 4145, 4127, 7198, 1583, 6119, 692, 1100, 3670, 2926, 5958, 2921, 5595, 970, 9509, 2416, 796, 3871, 2420, 8545, 6304, 9218, 8269, 514, 1665, 8965, 9118, 9774, 6677, 5668, 4508, 9520, 4385, 4550, 8040, 4153, 9950, 1515, 7359, 1459, 1049, 4310, 9969, 1931, 5987, 2313, 8792, 4362, 887, 2445, 8336, 3929, 4065, 6943, 1746, 5427, 6331, 5170, 9183, 1465, 5561, 3723, 4095, 9204, 2451, 7540, 1704, 2904, 883, 8486, 9633, 3792, 5925, 2387, 5510, 7278, 7126, 6120, 4933, 100, 8228, 4396, 7135, 2629, 6056, 2104, 9590, 3224, 5809, 7747, 6318, 6502, 4605, 7687, 8121, 2706, 6077, 2329, 2248, 5619, 3237, 5322, 3855, 4805, 8656, 7996, 7221, 3962, 3188, 7410, 9816, 1529, 3931, 9340, 7476, 8848, 8330, 5103, 6701, 6233, 8178, 9386, 5409, 9045, 1993, 3151, 8718, 3807, 842, 6746, 3759, 7166, 9598, 7605, 2835, 6401, 2817, 6394, 1707, 1580, 499, 9604, 3382, 6532, 9396, 1032, 8079, 9412, 7315, 9675, 6339, 2692, 7456, 8445, 7011, 4190, 1161, 1314, 7112, 1887, 237, 9717, 1566, 5718, 6577, 7970, 1925, 8698, 669, 2391, 7017, 2031, 4014, 145, 1761, 423, 4021, 7064, 2674, 7255, 2075, 5093, 1660, 5398, 5196, 6311, 1257, 5894, 3531, 7978, 929, 7421, 1745, 7066, 3219, 1842, 9784, 9149, 1080, 7363, 3748, 2822, 9494, 459, 8509, 4334, 9518, 5123, 6425, 4691, 1304, 1843, 10117, 6370, 3581, 4514, 2180, 6631, 6138, 8231, 6484, 6648, 8810, 4659, 9793, 8264, 7318, 3234, 2575, 258, 2344, 2750, 7284, 9488, 2114, 3613, 1172, 3056, 2224, 8019, 9750, 8059, 6305, 7056, 5819, 1252, 3592, 7806, 453, 479, 9634, 2015, 4536, 9253, 5463, 1130, 2865, 921, 5278, 616, 3089, 4654, 7354, 4664, 3127, 2253, 7742, 1177, 4671, 4048, 6861, 3788, 1877, 5023, 5253, 7061, 6740, 6308, 538, 9228, 3560, 1054, 4546, 5730, 6231, 3880, 5772, 286, 8033, 7445, 6144, 254, 1922, 6917, 1140, 2178, 425, 4793, 2928, 6836, 3785, 5867, 366, 4719, 8559, 2266, 8287, 623, 4900, 4383, 8212, 3018, 7715, 2267, 7451, 4527, 3844, 7947, 1174, 9999, 7026, 6603, 3620, 8724, 6627, 5296, 9523, 6931, 3116, 1048, 1090, 6008, 10102, 8964, 4165, 5358, 2302, 5997, 9794, 9106, 8338, 3047, 5687, 9248, 7046, 4114, 4571, 1343, 6378, 3242, 8183, 4579, 10081, 2998, 1308, 5556, 3762, 2148, 851, 9217, 8678, 6696, 9700, 1453, 2384, 9958, 1058, 4221, 1866, 6998, 8491, 427, 6134, 2487, 3985, 9631, 2014, 6244, 7911, 2438, 8206, 9963, 2870, 2083, 1913, 9235, 8810, 9331, 9431, 46, 964, 544, 1354, 9822, 3130, 6733, 4146, 2694, 4801, 10084, 7262, 4853, 6380, 7842, 10015, 880, 1617, 8385, 1568, 8760, 3687, 3492, 10037, 4307, 38, 2066, 7187, 2517, 373, 6178, 9975, 9469, 6127, 9481, 9768, 2832, 8320, 6810, 6901, 2122, 1421, 7450, 8386, 2328, 6362, 19, 6728, 2966, 3628, 2035, 5355, 5228, 9778, 7229, 3966, 10143, 8476, 7840, 4118, 156, 4598, 329, 8046, 1095, 7238, 7308, 3443, 2946, 2059, 3264, 8232, 2537, 5477, 1897, 3812, 4746, 3136, 8443, 7607, 5830, 8203, 7586, 7903, 7412, 1419, 3514, 4467, 2598, 6980, 5501, 3563, 6579, 8131, 8580, 5442, 6609, 4027, 8906, 3184, 8179, 6023, 2320, 2599, 8931, 4041, 7072, 8810, 6271, 9880, 8503, 3569, 2860, 3429, 860, 9238, 8260, 8983, 3437, 4666, 2303, 9500, 4827, 10056, 6232, 8957, 8360, 4101, 9436, 1851, 9503, 6, 91, 6614, 7155, 3308, 1999, 2887, 3181, 7481, 6260, 4897, 5177, 2333, 9589, 1191, 1312, 4614, 9056, 4989, 1045, 4669, 7179, 4266, 8959, 7397, 8298, 5921, 8238, 3253, 5338, 1879, 9221, 7774, 3496, 8602, 5817, 4445, 5086, 9770, 3228, 200, 2299, 9265, 491, 5225, 1107, 1599, 7259, 1655, 4250, 765, 1268, 2033, 4730, 7274, 666, 4001, 4959, 3409, 6368, 6490, 8074, 6624, 1701, 8271, 5202, 7264, 2581, 3447, 4111, 6711, 1684, 8925, 4487, 8767, 9212, 6621, 9161, 8810, 540, 9869, 2660, 9980, 4473, 7248, 6113, 9588, 8302, 4775, 2766, 8638, 6265, 8244, 4083, 4073, 4282, 4233, 3167, 967, 2450, 9919, 476, 2932, 9068, 6939, 5404, 1347, 6209, 3934, 6487, 888, 537, 8913, 6799, 7004, 1730, 2535, 7355, 885, 7683, 5453, 4237, 1817, 2434, 1282, 4613, 905, 6180, 7459, 3624, 6042, 9755, 3817, 7811, 6395, 4710, 10064, 1528, 8333, 3098, 8808, 8499, 3999, 9141, 688, 2906, 6714, 7541, 9716, 7244, 6216, 7270, 6438, 9762, 1280, 288, 3782, 1855, 5389, 7354, 2746, 9862, 9573, 5424, 9239, 5411, 4194, 2222, 8758, 2396, 3885, 6870, 7743, 3646, 7731, 5233, 1759, 4772, 8810, 2048, 5829, 8627, 4833, 9197, 3303, 9518, 6958, 1510, 2088, 9613, 9628, 9355, 2325, 1531, 2375, 7526, 7371, 8382, 6867, 3673, 8100, 932, 4060, 8734, 1290, 9551, 9146, 768, 738, 1880, 523, 7991, 2590, 5571, 2125, 4474, 6483, 1411, 4822, 488, 10164, 6014, 4994, 7839, 1332, 4696, 5146, 10010, 1549, 5020, 8270, 211, 7637, 1354, 7521, 648, 2288, 2007, 1555, 7441, 277, 2541, 263, 10143, 7046, 8041, 9624, 5637, 7514, 5682, 3567, 6680, 9149, 980, 3487, 2771, 2604, 1283, 6871, 6509, 6646, 10062, 2168, 5343, 3304, 4208, 6815, 7376, 5136, 10099, 7734, 2594, 8237, 5779, 7462, 6902, 6587, 7081, 8816, 8810, 823, 923, 2934, 2757, 424, 2965, 3689, 7269, 41, 7741, 6067, 3613, 1379, 8600, 3655, 8793, 1305, 8729, 1167, 6976, 7109, 4983, 6570, 2226, 3273, 8180, 8596, 5033, 6849, 2224, 1455, 2491, 8043, 3357, 6897, 1729, 4662, 1869, 222, 8205, 7025, 4348, 7818, 7930, 3998, 2053, 1913, 3004, 2213, 4036, 8363, 6374, 8871, 4726, 4367, 4211, 4219, 5077, 1734, 8647, 3485, 3293, 3259, 7900, 6274, 2852, 6557, 1203, 5813, 3097, 6267, 4980, 9864, 8636, 8119, 1520, 770, 8571, 7082, 3786, 6982, 6464, 6424, 1133, 5093, 9449, 5466, 9409, 2284, 565, 2606, 8169, 1968, 4794, 2010, 4499, 8375, 3997, 5365, 4679, 8810, 6130, 9436, 3488, 9765, 4167, 8904, 4717, 6085, 7437, 1686, 9904, 179, 4948, 7485, 7908, 9563, 8380, 4642, 36, 2399, 3588, 6402, 5674, 3945, 9304, 3957, 5839, 8252, 1195, 5008, 9478, 6225, 108, 9391, 4653, 3335, 519, 3431, 798, 9085, 4033, 6234, 9469, 5848, 9314, 9370, 1241, 2275, 5293, 3590, 1392, 284, 4501, 8622, 1711, 3774, 1134, 9272, 3190, 479, 1639, 9049, 1775, 5690, 6324, 5440, 7684, 9775, 2120, 6171, 7825, 8937, 448, 383, 8948, 4025, 7555, 4446, 701, 7433, 4957, 8952]
)



###   SCRIPT   ###
# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))


if idxs_train is not None:
    assert num_train == len(idxs_train)

def main():
    os.chdir(save_dir)
    
    train = mbGDMLTrain()
    train.load_dataset(dataset_path)
    train.train(
        model_name, num_train, num_validate, num_test, solver='analytic',
        sigmas=sigmas, save_dir='.', use_sym=True, use_E=True,
        use_E_cstr=use_E_cstr, use_cprsn=False, idxs_train=idxs_train,
        max_processes=None, overwrite=overwrite, torch=torch,
    )

main()