#pragma once

namespace madgrid {
  
enum class Domain {
  City,
  Game,
  Random,
  Warehouse
};

enum class Agents {
  Random20
};

enum class Tasks {
  Tasks20
};

struct TaskCtr {
  uint32_t ctr;
};

enum class TaskAssignmentStrategy {
  Greedy,
  RoundRobin,
  RoundRobinFixed
};

}
