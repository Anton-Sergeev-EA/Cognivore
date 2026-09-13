// pybind11 bindings exposing cognivore::FlatIndex and cognivore::NSWIndex as
// the `cognivore._native` extension module.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "vector_index.hpp"

namespace py = pybind11;
using namespace cognivore;

namespace {
// std::vector<uint8_t> <-> py::bytes helpers. pybind11/stl.h would happily
// convert a std::vector<uint8_t> to/from a Python *list* of ints, but that's
// both slower and a less natural API for a binary blob than plain `bytes`.
py::bytes to_py_bytes(const std::vector<uint8_t>& buf) {
    return py::bytes(reinterpret_cast<const char*>(buf.data()), buf.size());
}

std::vector<uint8_t> from_py_bytes(const py::bytes& data) {
    std::string s = data; // py::bytes converts implicitly to std::string
    return std::vector<uint8_t>(s.begin(), s.end());
}
} // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Cognivore native vector index (C++ core, exposed via pybind11)";

    py::class_<SearchResult>(m, "SearchResult")
        .def_readonly("id", &SearchResult::id)
        .def_readonly("score", &SearchResult::score)
        .def("__repr__", [](const SearchResult& r) {
            return "<SearchResult id=" + std::to_string(r.id) + " score=" +
                   std::to_string(r.score) + ">";
        });

    py::class_<FlatIndex>(m, "FlatIndex")
        .def(py::init<size_t>(), py::arg("dim"))
        .def("add", &FlatIndex::add, py::arg("id"), py::arg("vector"))
        .def("search", &FlatIndex::search, py::arg("query"), py::arg("k"))
        .def("__len__", &FlatIndex::size)
        .def_property_readonly("dim", &FlatIndex::dim)
        .def("serialize", [](const FlatIndex& self) { return to_py_bytes(self.serialize()); })
        .def_static("deserialize", [](const py::bytes& blob) {
            return FlatIndex::deserialize(from_py_bytes(blob));
        }, py::arg("blob"));

    py::class_<NSWIndex>(m, "NSWIndex")
        .def(py::init<size_t, size_t, size_t>(), py::arg("dim"), py::arg("m") = 16,
             py::arg("ef_construction") = 200)
        .def("add", &NSWIndex::add, py::arg("id"), py::arg("vector"))
        .def("search", &NSWIndex::search, py::arg("query"), py::arg("k"), py::arg("ef") = 150)
        .def("__len__", &NSWIndex::size)
        .def_property_readonly("dim", &NSWIndex::dim)
        .def("serialize", [](const NSWIndex& self) { return to_py_bytes(self.serialize()); })
        .def_static("deserialize", [](const py::bytes& blob) {
            return NSWIndex::deserialize(from_py_bytes(blob));
        }, py::arg("blob"));
}
