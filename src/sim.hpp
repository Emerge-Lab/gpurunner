#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>

#include "init.hpp"

namespace madgrid {

class Engine;

enum class ExportID : uint32_t {
  Reset,
  Action,
  Pose,
  Reward,
  Done,
  Map,
  Task,
  NumExports
};

struct Reset {
    int32_t resetNow;
};

enum class Action : int32_t {
  Move=0,
  RotateClockwise=1,
  RotateCounterCockwise=2,
  Wait=3,
};

enum class Heading : int32_t {
  Up = 0,
  Right = 1,
  Down = 2,
  Left = 3,
};

struct Location {
    int32_t row;
    int32_t col;
};

struct Pose {
    Location location;
    Heading heading;
};

struct Reward {
    float r;
};

struct Done {
    float episodeDone;
};

struct CurStep {
    uint32_t step;
};

enum class CollisionState {
  None,
  OutOfBounds,
  WallCollision,
  VertexCollision,
  EdgeCollision
};

struct CurTask{
    int32_t task;
};


struct Agent : public madrona::Archetype<Reset, Action, Pose, Reward, Done,
                                         CurStep, CurTask, CollisionState> {};

struct Sim : public madrona::WorldBase {
    struct Config {
        uint32_t maxEpisodeLength;
        bool enableViewer;
    };

    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    static void setupTasks(madrona::TaskGraphBuilder &builder,
                           const Config &cfg);

    Sim(Engine &ctx, const Config &cfg, const WorldInit &init);

    static constexpr madrona::CountT maxAgentCount{100};

    EpisodeManager *episodeMgr;
    uint32_t maxEpisodeLength;
    madrona::Entity agents[maxAgentCount];
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
