from ashlar.scripts import ashlar
from ashlar import reg
import numpy as np
import matplotlib.pyplot as plt
​
c1r = reg.BioformatsReader('FASCC-13_cycle0.nd2')
ashlar.process_axis_flip(c1r, True, False)
c2r = reg.BioformatsReader('FASCC-13_cycle1.nd2')
ashlar.process_axis_flip(c2r, True, False)
​
c1e = reg.EdgeAligner(c1r, max_shift=30, filter_sigma=0, verbose=True)
c1e.run()
c21l = reg.LayerAligner(c2r, c1e, max_shift=30, filter_sigma=0, verbose=True)
c21l.run()
​
c21l.mosaic_shape = c1e.mosaic_shape
vmin = np.percentile(c1e.reader.thumbnail[c1e.reader.thumbnail > 0], 1)
​
reg.plot_edge_quality(c1e, img=np.log1p(c1e.reader.thumbnail), im_kwargs=dict(vmin=np.log1p(vmin)))
plt.gcf().suptitle(f"-m {c1e.max_shift} --filter-sigma {c1e.filter_sigma}", color='salmon')
# save plot to disk
plt.gcf().tight_layout()
plt.gcf().set_size_inches(16, 9)
plt.gcf().savefig(f'edge_quality-m{c1e.max_shift}_s{c1e.filter_sigma}.jpg', bbox_inches='tight', dpi=144)
​
reg.plot_layer_quality(c21l, img=np.log1p(c1e.reader.thumbnail), im_kwargs=dict(vmin=np.log1p(vmin)))
plt.gcf().suptitle(f"-m {c1e.max_shift} --filter-sigma {c1e.filter_sigma}", color='salmon')
# save plot to disk
plt.gcf().tight_layout()
plt.gcf().set_size_inches(16, 9)
plt.gcf().savefig(f'layer_quality-m{c1e.max_shift}_s{c1e.filter_sigma}.jpg', bbox_inches='tight', dpi=144)
​
​
# Inspect rotation
​
# using napari
from ashlar import viewer
v = viewer.view_edges(c1e, tiles=range(141, 146))
viewer.view_edges(c21l, tiles=range(141, 146), viewer=v)
​
# using imreg_dft
# https://pypi.org/project/imreg_dft/
import imreg_dft
for i in range(143, 151):
    print(imreg_dft.imreg._get_ang_scale(c21l.overlap(i)[1:], 0))
​
# In [80]: for i in range(143, 151):
#     ...:     print(imreg_dft.imreg._get_ang_scale(c21l.overlap(i)[1:], 0))
#     ...:
# (1.0000507033513355, 0.16342539077689366)
# (1.0001891723557386, 0.1684640597475493)
# (0.9997530781653406, 0.16884097372422957)
# (1.000237636015229, 0.1657715317512043)
# (1.000503527855453, 0.15850173196633932)
# (1.0004809408770123, 0.1665515428289268)
# (1.0003915674224064, 0.1838691158465906)
# (1.0002914071957174, 0.15521869283534784)