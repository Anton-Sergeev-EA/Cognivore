// pybind11 bindings exposing the native vector indexes as
// the `cognivore._native` extension module.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "hnsw_index.hpp"
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
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

std::vector<float> one_dimensional(const FloatArray& array) {
    if (array.ndim() != 1) throw py::value_error("expected a one-dimensional vector");
    return std::vector<float>(array.data(), array.data() + array.size());
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

    py::class_<HNSWIndex>(m, "HNSWIndex")
        .def(py::init<size_t, size_t, size_t, uint32_t>(), py::arg("dim"), py::arg("m") = 16,
             py::arg("ef_construction") = 200, py::arg("seed") = 42)
        .def("add", [](HNSWIndex& index, VectorId id, const FloatArray& array) {
            const auto vector = one_dimensional(array);
            const py::gil_scoped_release release;
            index.add(id, vector);
        }, py::arg("id"), py::arg("vector"), "Insert a finite vector under the given ID.")
        .def("search", [](const HNSWIndex& index, const FloatArray& array, size_t k, size_t ef) {
            const auto query = one_dimensional(array);
            const py::gil_scoped_release release;
            return index.search(query, k, ef);
        }, py::arg("query"), py::arg("k"), py::arg("ef") = 150,
           "Find up to k neighbors using cosine similarity.")
        .def("__len__", &HNSWIndex::size)
        .def_property_readonly("dim", &HNSWIndex::dim)
        .def_property_readonly("max_level", &HNSWIndex::max_level)
        .def("serialize", [](const HNSWIndex& index) {
            std::vector<uint8_t> bytes;
            {
                const py::gil_scoped_release release;
                bytes = index.serialize();
            }
            return to_py_bytes(bytes);
        })
        .def_static("deserialize", [](const py::bytes& blob) {
            const auto* data = reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(blob.ptr()));
            const auto size = static_cast<size_t>(PyBytes_GET_SIZE(blob.ptr()));
            const py::gil_scoped_release release;
            return HNSWIndex::deserialize(data, size);
        }, py::arg("blob"));
}
