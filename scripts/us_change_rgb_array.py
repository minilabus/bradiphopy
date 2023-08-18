from bradiphopy.bradipho_helper import BraDiPhoHelper3D
from bradiphopy.io import load_polydata, save_polydata

import sys
import numpy as np

polydata = load_polydata(sys.argv[1])
bdp = BraDiPhoHelper3D(polydata)
rgba = bdp.get_scalar('RGB')
new = BraDiPhoHelper3D.generate_bdp_obj(bdp.get_polydata_vertices(), bdp.get_polydata_triangles())
new.set_scalar(rgba[:, 0:3], 'RGB', dtype=np.uint8)
new.set_scalar(rgba[:, 0:3], 'RGB', dtype=np.uint8)
save_polydata(new.get_polydata(), sys.argv[2], ascii=False)