This file documents a python module built from C++ code with pybind11.
You should document the Python interfaces, *NOT* the C++ interfaces.

Module memcpyable.core
**************************

Module :py:mod:`core` built from C++ code in :file:`memcpyable/core/core.cpp`.

.. function:: add(x,y,z)
   :module: memcpyable.core
   
   Compute the sum of *x* and *y* and store the result in *z* (overwrite).

   :param x: 1D Numpy array with ``dtype=numpy.float64`` (input)
   :param y: 1D Numpy array with ``dtype=numpy.float64`` (input)
   :param z: 1D Numpy array with ``dtype=numpy.float64`` (output)
   