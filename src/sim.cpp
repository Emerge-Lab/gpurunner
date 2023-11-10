#include "sim.hpp"
#include <array>
#include <madrona/mw_gpu_entry.hpp>

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

    // Export tensors for pytorch
    registry.exportColumn<Agent, Reset>((uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, Pose>((uint32_t)ExportID::Pose);
    registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>((uint32_t)ExportID::Done);
}

inline void manageEpisode(Engine &ctx, Done &done, Reset &reset,
                          CurStep &episodeStep) {
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
        ++currentStep;
        return;
    }

    done.episodeDone = 1.f;
    done.episodeDone = 0.f;
    ++currentStep;
}

inline void performAction(Engine & /* unused */, Action &action, Pose &pose) {
    if (action == Action::Wait) {
        return;
    }

    if (action == Action::RotateClockwise ||
        action == Action::RotateCounterCockwise) {
        // TODO
    }

    assert(action == Action::Move);

    const std::array<Location, 4> headingToOffsets{
        {{-1, 0}, {0, 1}, {0, 1}, {-1, 0}}};
    auto &offset = headingToOffsets[static_cast<std::size_t>(pose.heading)];
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

void Sim::setupTasks(TaskGraphBuilder &builder, const Config &)
{
    auto actionSystem =
        builder
            .addToGraph<ParallelForNode<Engine, performAction, Action, Pose>>(
                {});

    // TODO: Compute CollisionState
    // TODO: Compute Reward
    // TODO: Manage episode

#ifdef MADRONA_GPU_MODE
    auto sort_agents = queueSortByWorld<Agent>(builder, {actionSystem});
    (void)sort_agents;
#else
    (void)actionSystem;
#endif
}

Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      grid(init.grid),
      maxEpisodeLength(cfg.maxEpisodeLength)
{
    Entity agent = ctx.makeEntity<Agent>();
    // TODO: Is there a need to introduce an Action::None so as to distinguish
    // between the environment issuing a WAIT and the learning component issuing
    // a WAIT
    ctx.get<Action>(agent) = Action::Wait;
    ctx.get<Reward>(agent).r = 0.f;
    ctx.get<Done>(agent).episodeDone = 0.f;
    ctx.get<CurStep>(agent).step = 0;
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
