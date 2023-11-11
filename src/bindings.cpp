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


NB_MODULE(_gridworld_madrona, m) {
    madrona::py::setupMadronaSubmodule(m);

    nb::class_<Manager> (m, "GridWorldSimulator")
        .def("__init__", [](Manager *self,
                            int64_t max_episode_length,
                            madrona::py::PyExecMode exec_mode,
                            int64_t num_worlds,
                            int64_t gpu_id) {
            new (self) Manager(Manager::Config {
                .maxEpisodeLength = (uint32_t)max_episode_length,
                .execMode = exec_mode,
                .numWorlds = (uint32_t)num_worlds,
                .gpuID = (int)gpu_id,
	      });
        }, nb::arg("max_episode_length"),
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
