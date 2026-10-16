#define PY_SSIZE_T_CLEAN
#include <Python.h>
static PyObject *noop(PyObject *self, PyObject *unused) { Py_RETURN_NONE; }
static PyObject *add3(PyObject *self, PyObject *const *args, Py_ssize_t n) {
    if (n != 3) { PyErr_SetString(PyExc_TypeError, "add3 takes 3 floats"); return NULL; }
    double a = PyFloat_AsDouble(args[0]), b = PyFloat_AsDouble(args[1]), c = PyFloat_AsDouble(args[2]);
    if (PyErr_Occurred()) return NULL;
    return PyFloat_FromDouble(a + b + c);
}
static PyMethodDef methods[] = {
    {"noop", noop, METH_NOARGS, NULL},
    {"add3", (PyCFunction)(void (*)(void))add3, METH_FASTCALL, NULL},
    {NULL, NULL, 0, NULL}};
static struct PyModuleDef mod = {PyModuleDef_HEAD_INIT, "capi_mod", NULL, -1, methods};
PyMODINIT_FUNC PyInit_capi_mod(void) { return PyModule_Create(&mod); }
