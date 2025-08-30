//
// Created by Shlok Jain on 30/08/25.
//

#ifndef GRAPH_CONSTANTS_H
#define GRAPH_CONSTANTS_H

// Edge use. Indicates specialized uses.
// Maximum value that can be stored for a directed edge is 63 - DO NOT EXCEED!
enum class Use : uint8_t {
    // Road specific uses
    kRoad = 0,
    kRamp = 1,            // Link - exits/entrance ramps.
    kTurnChannel = 2,     // Link - turn lane.
    kTrack = 3,           // Agricultural use, forest tracks
    kDriveway = 4,        // Driveway/private service
    kAlley = 5,           // Service road - limited route use
    kParkingAisle = 6,    // Access roads in parking areas
    kEmergencyAccess = 7, // Emergency vehicles only
    kDriveThru = 8,       // Commercial drive-thru (banks/fast-food)
    kCuldesac = 9,        // Cul-de-sac (edge that forms a loop and is only
                          // connected at one node to another edge.
    kLivingStreet = 10,   // Streets with preference towards bicyclists and pedestrians
    kServiceRoad = 11,    // Generic service road (not driveway, alley, parking aisle, etc.)

    // Bicycle specific uses
    kCycleway = 20,     // Dedicated bicycle path
    kMountainBike = 21, // Mountain bike trail

    kSidewalk = 24,

    // Pedestrian specific uses
    kFootway = 25,
    kSteps = 26, // Stairs
    kPath = 27,
    kPedestrian = 28,
    kBridleway = 29,
    kPedestrianCrossing = 32, // cross walks
    kElevator = 33,
    kEscalator = 34,
    kPlatform = 35,

    // Rest/Service Areas
    kRestArea = 30,
    kServiceArea = 31,

    // Other... currently, either BSS Connection or unspecified service road
    kOther = 40,

    // Ferry and rail ferry
    kFerry = 41,
    kRailFerry = 42,

    kConstruction = 43, // Road under construction

    // Transit specific uses. Must be last in the list
    kRail = 50,               // Rail line
    kBus = 51,                // Bus line
    kEgressConnection = 52,   // Connection egress <-> station
    kPlatformConnection = 53, // Connection station <-> platform
    kTransitConnection = 54,  // Connection osm <-> egress
  };

enum class Surface : uint8_t {
    kPavedSmooth = 0,
    kPaved = 1,
    kPavedRough = 2,
    kCompacted = 3,
    kDirt = 4,
    kGravel = 5,
    kPath = 6,
    kImpassable = 7
  };

#endif //GRAPH_CONSTANTS_H
