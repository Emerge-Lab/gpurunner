#pragma once

#include "sim.hpp"

namespace madgrid {
  
Location fromRowMajor(int32_t linearized, int32_t rowCount, int32_t colCount) {
  int32_t col = linearized % colCount;
  int32_t row = (int32_t)(linearized / colCount);

  return Location { .row = row, .col = col };
}

}
