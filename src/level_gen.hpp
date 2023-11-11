#pragma once

#include "sim.hpp"

namespace madgrid {

void createPersistentEntities(Engine &ctx);

// First, destroys any non-persistent state for the current world and then
// generates a new play area.
void generateWorld(Engine &ctx);

}