from cffi import FFI
ffi = FFI()
ffi.cdef("void noop(void); double add3(double, double, double);")
ffi.set_source("cffi_api_mod", "void noop(void) {} double add3(double a, double b, double c) { return a + b + c; }")
ffi.compile(verbose=False)
