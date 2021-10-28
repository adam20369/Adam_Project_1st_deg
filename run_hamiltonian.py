from PXP_TI_Bath import *
import sys

n = int(sys.argv[1])

H = PXPOBCNew(n)

np.save('hdir/pxp_obc_n_%.npy' % n, H)