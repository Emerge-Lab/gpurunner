#include "sim.hpp"
#include <array>
#include <madrona/mw_gpu_entry.hpp>
#include "level_gen.cpp"

#include <iostream>

using namespace madrona;
using namespace madrona::math;

namespace madgrid {

void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);

    registry.registerComponent<Reset>();
    registry.registerComponent<Action>();
    registry.registerComponent<Pose>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<CurStep>();
    registry.registerComponent<CollisionState>();

    registry.registerArchetype<Agent>();

    registry.registerSingleton<Map>();

    // Export tensors for pytorch
    registry.exportColumn<Agent, Reset>((uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, Pose>((uint32_t)ExportID::Pose);
    registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>((uint32_t)ExportID::Done);

    registry.exportSingleton<Map>((uint32_t)ExportID::Map);
}

inline void manageEpisode(Engine &ctx, Done &done, Reset &reset,
                          CurStep &episodeStep) {
    printf("Reset issued\n");
    bool episodeDone{false};

    if (reset.resetNow != 0) {
        reset.resetNow = 0;
        episodeDone = true;
    }

    auto &currentStep = episodeStep.step;

    if (currentStep == ctx.data().maxEpisodeLength - 1) {
        episodeDone = true;
    }

    if (not episodeDone) {
        done.episodeDone = 0.f;
        ++currentStep;
        return;
    }


    done.episodeDone = 1.f;
    currentStep = 0;

    generateWorld(ctx);
}

inline void performAction(Engine & /* unused */, Action &action, Pose &pose) {
  printf("action=");
    switch (action) {
    case Action::Move:
      printf("move");
      break;
    case Action::RotateClockwise:
      printf("rotate lcockwise");
      break;
    case Action::RotateCounterCockwise:
      printf("rotate counter clockwise");
      break;
    case Action::Wait:
      printf("wait");
      break;
    default:
      printf("none");
    }

    printf("\n");
  
  printf("pose.heading=");
    switch (pose.heading) {
    case Heading::Up:
      printf("up");
      break;
    case Heading::Right:
      printf("right");
      break;
    case Heading::Down:
      printf("down");
      break;
    case Heading::Left:
      printf("left");
      break;
    default:
      printf("none");
    }

    printf("\n");
	   
    if (action == Action::Wait) {
        return;
    }

    if (action == Action::RotateClockwise) {
      switch (pose.heading) {
      case Heading::Up:
	pose.heading = Heading::Right;
	break;
      case Heading::Right:
	pose.heading = Heading::Down;
	break;
      case Heading::Down:
	pose.heading = Heading::Left;
	break;
      case Heading::Left:
	pose.heading = Heading::Up;
	break;
      }

      return;
    }

    if (action == Action::RotateCounterCockwise) {
      switch (pose.heading) {
      case Heading::Up:
	pose.heading = Heading::Left;
	break;
      case Heading::Right:
	pose.heading = Heading::Up;
	break;
      case Heading::Down:
	pose.heading = Heading::Right;
	break;
      case Heading::Left:
	pose.heading = Heading::Down;
	break;
      }

      return;
    }

    assert(action == Action::Move);

    const std::array<Location, 4> headingToOffsets{
        {{-1, 0}, {0, 1}, {0, 1}, {-1, 0}}};

    
    //    printf("heading %d\n", static_cast<int32_t>(pose.heading));
    
    auto &offset = headingToOffsets[static_cast<int32_t>(pose.heading)];
    pose.location.row += offset.row;
    pose.location.col += offset.col;
}

#ifdef MADRONA_GPU_MODE
template <typename ArchetypeT>
TaskGraph::NodeID queueSortByWorld(TaskGraph::Builder &builder,
                                   Span<const TaskGraph::NodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(
            deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}
#endif

inline void debug(Engine & /* unused */, Pose &pose) {
    printf("heading %d\n", static_cast<int32_t>(pose.heading));
}

void Sim::setupTasks(TaskGraphBuilder &builder, const Config &)
{
    auto actionSystem =
        builder
            .addToGraph<ParallelForNode<Engine, performAction, Action, Pose>>(
                {});

    // auto dbgSystem = builder.addToGraph<ParallelForNode<Engine, debug, Pose>>({actionSystem});
    auto episodeSystem = builder.addToGraph<ParallelForNode<Engine, manageEpisode, Done, Reset, CurStep>>({actionSystem});

    // TODO: Compute CollisionState
    // TODO: Compute Reward

#ifdef MADRONA_GPU_MODE
    auto sort_agents = queueSortByWorld<Agent>(builder, {episodeSystem});
    (void)sort_agents;
#else
    (void)episodeSystem;
#endif
}

Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      maxEpisodeLength(cfg.maxEpisodeLength) {
    createPersistentEntities(ctx, init.domain, init.agents, init.tasks);
    generateWorld(ctx);
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
