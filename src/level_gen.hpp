#pragma once

#include "sim.hpp"
#include "types.hpp"

namespace madgrid {

void createPersistentEntities(Engine &ctx, Domain domain, Agents agents, Tasks tasks);

// First, destroys any non-persistent state for the current world and then
// generates a new play area.
void generateWorld(Engine &ctx);

}
