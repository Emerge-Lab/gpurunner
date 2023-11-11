#pragma once

namespace madgrid {

enum class CellFlag : int {
    FreeSpace,
    HardObstacle,
    Tree,
    Emitter,
    Source
};

struct Cell {
    CellFlag flags;
};

struct GridState {
    Cell *cells;
    int32_t width;
    int32_t height;
};

struct Map {
    int32_t map[1024];
};

}
