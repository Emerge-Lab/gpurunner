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

enum class TaskAssignmentStrategy {
  Greedy,
  RoundRobin,
  RoundRobinFixed
};

}
