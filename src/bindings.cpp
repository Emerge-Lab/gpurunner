#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

namespace madgrid {

template <typename T>
static void setRewards(Cell *cells,
                       T *rewards,
                       int64_t grid_x,
                       int64_t grid_y)
{
    // for (int64_t y = 0; y < grid_y; y++) {
    //     for (int64_t x = 0; x < grid_x; x++) {
    //         int64_t idx = y * grid_x + x;
    //         cells[idx].reward = static_cast<float>(rewards[idx]);
    //     }
    // }
}


static Cell * setupCellData(
    const nb::ndarray<void, nb::shape<nb::any, nb::any>,
        nb::c_contig, nb::device::cpu> &walls,
    const nb::ndarray<void, nb::shape<nb::any, nb::any>,
        nb::c_contig, nb::device::cpu> &rewards,
    const nb::ndarray<void, nb::shape<nb::any, 2>,
        nb::c_contig, nb::device::cpu> &end_cells,
    int64_t grid_x,
    int64_t grid_y)

{
    Cell *cells = new Cell[grid_x * grid_y]();
    
    if (rewards.dtype() == nb::dtype<float>()) {
        setRewards(cells, (float *)rewards.data(), grid_x, grid_y);
    } else if (rewards.dtype() == nb::dtype<double>()) {
        setRewards(cells, (double *)rewards.data(), grid_x, grid_y);
    } else {
        throw std::runtime_error("rewards: unsupported input type");
    }

    return cells;
}

NB_MODULE(_gridworld_madrona, m) {
    madrona::py::setupMadronaSubmodule(m);

    nb::class_<Manager> (m, "GridWorldSimulator")
        .def("__init__", [](Manager *self,
                            nb::ndarray<void, nb::shape<nb::any, nb::any>,
                                nb::c_contig, nb::device::cpu> walls,
                            nb::ndarray<void, nb::shape<nb::any, nb::any>,
                                nb::c_contig, nb::device::cpu> rewards,
                            nb::ndarray<void, nb::shape<nb::any, 2>,
                                nb::c_contig, nb::device::cpu> end_cells,
                            int64_t start_x,
                            int64_t start_y,
                            int64_t max_episode_length,
                            madrona::py::PyExecMode exec_mode,
                            int64_t num_worlds,
                            int64_t gpu_id) {
            int64_t grid_y = (int64_t)walls.shape(0);
            int64_t grid_x = (int64_t)walls.shape(1);

            if ((int64_t)rewards.shape(0) != grid_y ||
                (int64_t)rewards.shape(1) != grid_x) {
                throw std::runtime_error("walls and rewards shapes don't match");
            }

            Cell *cells =
                setupCellData(walls, rewards, end_cells, grid_x, grid_y);

            new (self) Manager(Manager::Config {
                .maxEpisodeLength = (uint32_t)max_episode_length,
                .execMode = exec_mode,
                .numWorlds = (uint32_t)num_worlds,
                .gpuID = (int)gpu_id,
	      });

            delete[] cells;
        }, nb::arg("walls"),
           nb::arg("rewards"),
           nb::arg("end_cells"),
           nb::arg("start_x"),
           nb::arg("start_y"),
           nb::arg("max_episode_length"),
           nb::arg("exec_mode"),
           nb::arg("num_worlds"),
           nb::arg("gpu_id") = -1)
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("map_tensor", &Manager::mapTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("observation_tensor", &Manager::observationTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("jax", madrona::py::JAXInterface::buildEntry<
                &Manager::trainInterface,
                &Manager::step
#ifdef MADRONA_CUDA_SUPPORT
                , &Manager::gpuRolloutStep
#endif
            >())
    ;
}

}
