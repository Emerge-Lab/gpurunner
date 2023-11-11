#pragma once

#include "sim.hpp"

namespace madgrid {
  
Location fromRowMajor(int32_t linearized, int32_t rowCount, int32_t colCount) {
  int32_t row = linearized % rowCount;
  int32_t col = linearized - row * rowCount;

  return Location { .row = row, .col = col };
}

}
