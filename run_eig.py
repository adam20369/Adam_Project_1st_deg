from PXP_TI_Bath import *
import os
import sys

n = int(sys.argv[1])

if os.path.isfile('hdir/pxp_obc_n_%i.npy' % n):
    H = np.load('hdir/pxp_obc_n_%i.npy' % n)
else:
    H = PXPOBCNew(n)
    np.save('hdir/pxp_obc_n_%i.npy' % n, H)

E = la.eigvalsh(H)

np.save('spectrum/pxp_obc_n_%i.npy' % n, E)