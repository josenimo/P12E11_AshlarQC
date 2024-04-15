#user1
from ashlar import fileseries, reg
import matplotlib.pyplot as plt
from ashlar.scripts.ashlar import process_axis_flip

c1r = fileseries.FileSeriesReader(
    '/run/media/jmamede/Joao/multiplex/ENDO/img1/',
    pattern='ENDO-cGAMP_647-YB1_TRITC-pagGFP-DAPI-paGFP_v{series:170}_PRJ.Custom.ome.tiff',
    overlap=0.05,
    width=17,
    height=10,
    layout='snake',
    direction='vertical',
    pixel_size=0.10833,
)

#yuanchen responds
from ashlar import fileseries, thumbnail
import matplotlib.pyplot as plt
from ashlar.scripts.ashlar import process_axis_flip
import numpy as np

# if you know the path of you stage when scaning the sample
# choose `layout='snake'` or `layout='raster'`
c1r = fileseries.FileSeriesReader(
    '/run/media/jmamede/Joao/multiplex/ENDO/img1/',
    pattern='ENDO-cGAMP_647-YB1_TRITC-pagGFP-DAPI-paGFP_v{series}_PRJ.Custom.ome.tiff',
    overlap=0.05,
    width=17,
    height=10,
    layout='snake',
    direction='vertical',
    pixel_size=0.10833,
)
# check if filenames are indexed properly
print(
    '\n'.join(c1r.metadata.filename(i, 0) for i in range(80, 120))
)
# it's very likely you'll want `flip_x=False, flip_y=True`
# try not to tweak here for now
process_axis_flip(c1r, flip_x=False, flip_y=True)
# If using the third channel, pass `channel=2` (0-based indexing in python)
thumbnail_c1r = thumbnail.make_thumbnail(c1r, channel=2)

plt.figure()
# doing log just for visualization
plt.imshow(np.log(thumbnail_c1r))