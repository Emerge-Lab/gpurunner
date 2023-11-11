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
  NumExports,
};

struct Reset {
    int32_t resetNow;
};

enum class Action : int32_t {
  Move,
  RotateClockwise,
  RotateCounterCockwise,
  Wait,
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
  OutOfBounds,
  WallCollision,
  VertexCollision,
  EdgeCollision
};


struct Agent : public madrona::Archetype<Reset, Action, Pose, Reward, Done,
                                         CurStep, CollisionState> {};

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

    EpisodeManager *episodeMgr;
    uint32_t maxEpisodeLength;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
