#include "level_gen.hpp"
#include <cassert>
#include <cmath>
// #include <stdlib.h>
// #include <iostream>

namespace {
     char arr[] = {'.','.','.','.','.','.','.','.','.','.','@','.','.','.','.','.','.','@','.','.','.','@','.','@','.','.','.','.','.','.','.','.','@','.','.','.','@','.','@','@','.','.','.','.','.','.','.','.','.','.','.','@','.','@','.','.','.','@','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','@','.','.','.','.','.','.','.','.','@','@','.','.','.','@','.','.','.','.','.','.','.','.','@','.','@','.','.','.','.','.','.','.','.','.','.','@','.','.','.','.','.','.','.','.','.','.','@','.','.','.','@','@','.','.','.','.','.','.','.','.','.','.','.','@','@','.','@','.','.','.','.','.','.','@','.','.','@','.','.','.','.','.','.','.','.','.','.','@','.','.','@','.','@','.','.','@','@','.','.','.','.','.','.','@','.','.','.','.','@','.','.','@','@','.','.','.','.','.','.','@','.','.','@','@','.','.','.','.','.','.','.','.','@','.','.','.','.','.','.','.','@','.','.','.','@','.','@','.','.','.','@','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','@','@','.','.','.','@','.','.','@','.','.','.','.','.','.','.','.','@','.','.','.','.','.','.','.','.','.','@','.','.','.','.','.','.','.','@','.','@','.','.','.','.','.','.','.','@','.','@','.','.','.','.','.','.','.','.','.','.','.','.','@','.','@','.','.','.','.','.','.','.','@','@','.','.','.','@','.','@','.','.','.','.','.','.','.','.','.','@','.','@','.','.','.','.','.','.','.','@','.','.','.','@','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','@','.','.','.','.','.','@','.','@','.','@','.','.','.','.','.','.','.','@','.','.','.','@','.','.','.','.','.','@','.','@','.','@','@','.','@','.','.','.','.','.','.','.','.','.','.','@','.','.','.','.','.','.','@','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','@','.','@','.','.','.','.','.','@','.','.','.','@','.','.','.','.','@','@','@','.','.','.','.','.','.','.','@','.','.','.','.','.','.','.','.','.','@','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','@','.','.','.','.','@','.','@','.','@','.','.','@','.','.','.','.','.','.','.','.','.','.','.','.','.','@','.','.','.','@','.','.','.','.','.','.','.','.','@','.','.','.','.','@','.','.','@','@','@','.','.','.','.','@','.','.','@','.','.','.','.','.','.','@','@','@','.','.','@','.','@','.','.','@','.','@','.','.','@','@','.','.','.','.','@','T','@','@','.','.','.','.','.','.','.','.','.','@','.','.','.','.','.','.','.','.','@','.','.','@','.','.','.','.','.','.','@','@','@','.','.','.','.','.','.','@','.','.','.','@','.','.','@','.','.','@','.','@','.','.','.','@','.','.','.','@','.','.','.','.','.','.','@','.','@','.','.','.','.','.','.','.','.','@','.','.','.','.','.','.','.','.','.','@','@','.','@','.','.','.','@','.','.','.','.','.','.','.','.','.','@','.','@','.','.','.','.','.','.','.','.','@','.','.','@','@','@','.','.','@','@','@','.','.','.','.','.','@','.','@','.','.','@','.','.','.','.','@','.','.','.','.','.','.','@','.','.','.','.','.','.','.','.','.','.','@','.','.','.','.','.','.','.','.','@','.','.','.','.','.','@','.','.','.','.','@','.','.','.','@','.','@','.','@','@','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','@','.','.','.','@','.','@','.','.','@','.','@','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','@','.','.','.','.','.','.','.','.','@','.','.','@','.','.','@','.','@','.','.','.','.','.','.','@','.','@','.','.','.','.','.','.','.','.','.','@','@','.','.','.','.','.','.','.','.','.','@','.','.','@','.','.','@','.','.','@','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','@','.','.','@','@','.','.','.','.','.','.','@','.','.','.','@','.','.','.','@','.','.','.','.','.','.','.','.','.','@','@','@','@','.','.','.','.','@','.','.','.','.','@','.','.','.','.','@','.','@','.','@','.','.','.','@','.','.','.','.','.','.','@','@','@','@','.','@','.','.','@','@','.','@','.','@','@','.','.','.','.','.','@','.','.','.','.','.','.','.','.','@','.','.','.','.','@','@','.','.','.','.','.','.','.','.','.','.','.','.','.','@','.','.','@','.','.','.','.','.','.','.','.','.','@','.','.','.'};

}

namespace madgrid {

using namespace madrona;
using namespace madrona::math;

// Register the entity with the broadphase system
// This is needed for every entity with all the physics components.
// Not registering an entity will cause a crash because the broadphase
// systems will still execute over entities with the physics components.
// static void registerRigidBodyEntity(
//     Engine &ctx,
//     Entity e,
//     SimObject sim_obj)
// {
//     ObjectID obj_id { (int32_t)sim_obj };
//     ctx.get<broadphase::LeafID>(e) =
//         RigidBodyPhysicsSystem::registerEntity(ctx, e, obj_id);
// }


// float degreesToRadians(float degrees) { return degrees * M_PI / 180.0; }

static inline void resetAgent(Engine &ctx, Entity agent) {
    // auto xCoord = ctx.get<Trajectory>(vehicle).positions[0].x;
    // auto yCoord = ctx.get<Trajectory>(vehicle).positions[0].y;
    // auto xVelocity = ctx.get<Trajectory>(vehicle).velocities[0].x;
    // auto yVelocity = ctx.get<Trajectory>(vehicle).velocities[0].y;
    // auto speed = ctx.get<Trajectory>(vehicle).velocities[0].length();
    // auto heading = ctx.get<Trajectory>(vehicle).initialHeading;

    // ctx.get<BicycleModel>(vehicle) = {
    //     .position = {.x = xCoord, .y = yCoord}, .heading = heading, .speed = speed};
    // ctx.get<Position>(vehicle) = Vector3{.x = xCoord, .y = yCoord, .z = 1};
    // ctx.get<Rotation>(vehicle) = Quat::angleAxis(heading, madrona::math::up);
    // ctx.get<Velocity>(vehicle) = {
    //     Vector3{.x = xVelocity, .y = yVelocity, .z = 0}, Vector3::zero()};

    // ctx.get<ExternalForce>(vehicle) = Vector3::zero();
    // ctx.get<ExternalTorque>(vehicle) = Vector3::zero();
    // ctx.get<Action>(vehicle) =
    //     Action{.acceleration = 0, .steering = 0, .headAngle = 0};
    // ctx.get<StepsRemaining>(vehicle).t = consts::episodeLen;
}

// static inline Entity createAgent(Engine &ctx, float xCoord, float yCoord,
//                                    float length, float width, float heading,
//                                    float speedX, float speedY, int32_t idx) {
//     auto vehicle = ctx.makeEntity<Agent>();

//     // The following components do not vary within an episode and so need only
//     // be set once
//     ctx.get<VehicleSize>(vehicle) = {.length = length, .width = width};
//     ctx.get<Scale>(vehicle) = Diag3x3{.d0 = width, .d1 = length, .d2 = 1};
//     ctx.get<ObjectID>(vehicle) = ObjectID{(int32_t)SimObject::Cube};
//     ctx.get<ResponseType>(vehicle) = ResponseType::Dynamic;
//     ctx.get<EntityType>(vehicle) = EntityType::Agent;

//     // Since position, heading, and speed may vary within an episode, their
//     // values are retained so that on an episode reset they can be restored to
//     // their initial values.
//     ctx.get<Trajectory>(vehicle).positions[0] =
//         Vector2{.x = xCoord, .y = yCoord};
//     ctx.get<Trajectory>(vehicle).initialHeading = degreesToRadians(heading);
//     ctx.get<Trajectory>(vehicle).velocities[0] =
//         Vector2{.x = speedX, .y = speedY};

//     // This is not stricly necessary since , but is kept here for consistency
//     resetVehicle(ctx, vehicle);

//     return vehicle;
// }

uint32_t Rand() {
    auto Seed = 42;
    uint32_t Val = Seed + 0x000b07a1;
    Seed = (Val * 0x3c7c0ac1);
    // Only lowest 19 bits are random-ish.
    return Seed & 0x7ffff;
}


void createPersistentEntities(Engine &ctx) {
    // std::ifstream data(pathToScenario);
    // assert(data.is_open());
    // using nlohmann::json;

    // json rawJson;
    // data >> rawJson;
    // std::ifstream file("/home/aarav/gpurunner/example_problems/city.domain/maps/Paris_1_256.map"); // Replace with your file path


    // std::random_device rd;
    // std::mt19937 rand_gen(rd());
    // std::uniform_int_distribution<int32_t> mapObstacleGen(0,4);
    // std::uniform_int_distribution<int32_t> IsObstacle(0,4);
    
   
    // ctx.singleton<Map>() = {
    //     .cells = new Cell[32*32],
    //     .width = 32,
    //     .height = 32
    // };

    for (int i =0 ; i<(32*32); i++)
    {
        if(arr[i] == '.')
            ctx.singleton<Map>().map[i] = (int32_t)CellFlag::FreeSpace;
        else if(arr[i] == '@')
            ctx.singleton<Map>().map[i] = (int32_t)CellFlag::HardObstacle;
        else if(arr[i] == 'T')
            ctx.singleton<Map>().map[i] = (int32_t)CellFlag::Tree;
        else if(arr[i] == 'E')
            ctx.singleton<Map>().map[i] = (int32_t)CellFlag::Emitter;
        else if(arr[i] == 'S')
            ctx.singleton<Map>().map[i] = (int32_t)CellFlag::Source;
    }
    // int t = 0;

    // auto numElems = 32*32;

    // for(int i =0 ; i < numElems; i++)
    // {
    //     auto r = Rand();
    //     auto obstacle = Rand();
    //     if(r<104857)
    //         ctx.data().grid->cells[i].flags = CellFlag::HardObstacle;
    //     else
    //         ctx.data().grid->cells[i].flags = CellFlag::FreeSpace;
    // }


    // std::string mapFile = data["mapFile"];
    // std::string agentFile = data["agentFile"];
    // std::string taskFile = data["taskFile"];



    // TODO(samk): handle keys not existing
    // size_t agentCount{0};


    // for (const auto &obj : rawJson["objects"]) {
    //   if (agentCount == consts::numAgents) {
    //     break;
    //   }
    //   if (obj["type"] != "vehicle") {
    //     continue;
    //   }
    //   auto vehicle = createVehicle(
    //       ctx,
    //       // TODO(samk): Nocturne allows for configuring the initial position
    //       // but in practice it looks to always be set to 0.
    //       obj["position"][0]["x"], obj["position"][0]["y"], obj["length"],
    //       obj["width"], obj["heading"][0], obj["velocity"][0]["x"],
    //       obj["velocity"][0]["y"], agentCount);

    //   ctx.data().agents[agentCount++] = vehicle;
    // }
}

static void resetPersistentEntities(Engine &ctx) {
}

static void generateLevel(Engine &ctx) {}

void generateWorld(Engine &ctx)
{
    resetPersistentEntities(ctx);
    generateLevel(ctx);
}

}