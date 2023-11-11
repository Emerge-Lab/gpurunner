#pragma once

#include <madrona/sync.hpp>

#include "grid.hpp"
#include "types.hpp"

namespace madgrid {

struct EpisodeManager {
    madrona::AtomicU32 curEpisode;
};

struct WorldInit {
    EpisodeManager *episodeMgr;
    Domain domain;
    Agents agents;
    Tasks tasks;
};

}
