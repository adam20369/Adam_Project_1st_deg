from PXP_TI_Bath import *
import os

spec_list = os.listdir('spectrum')
print(spec_list)
for spec in spec_list:
    print(spec)
    E = np.load('spectrum/%s' % spec)
    r = RMeanMetric(E)
    np.save('rmetric/%s' % spec, r)

