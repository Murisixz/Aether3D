#include <nanobind/nanobind.h>
NB_MODULE(nb_mod, m) {
    m.def("noop", []() {});
    m.def("add3", [](double a, double b, double c) { return a + b + c; });
}
