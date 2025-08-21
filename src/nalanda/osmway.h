//
// Created by Shlok Jain on 21/08/25.
//

#ifndef OSMWAY_H
#define OSMWAY_H

#pragma once
#include <cstdint>
#include <cstring>

namespace pollu {

    struct OSMWay {
        uint64_t osmid_;   // OSM way id
        uint16_t nodecount_;  // number of nodes in way

        // --- Indexes ---
        uint32_t name_idx_;
        uint32_t ref_idx_;

        // --- Classification ---
        uint32_t road_class_ : 3;  // trunk, primary, residential, footway...
        uint32_t use_        : 6;  // footway, steps, alley, cycleway, etc.
        uint32_t lanes_      : 4;
        uint32_t surface_    : 3;  // paved/unpaved/gravel/etc.

        // --- Routing flags ---
        uint32_t oneway_     : 1;
        uint32_t oneway_rev_ : 1;
        uint32_t roundabout_ : 1;
        uint32_t tunnel_     : 1;
        uint32_t bridge_     : 1;
        uint32_t indoor_     : 1;
        uint32_t lit_        : 1;

        uint32_t pedestrian_fwd_ : 1;
        uint32_t pedestrian_bwd_ : 1;
        uint32_t sidewalk_left_  : 1;
        uint32_t sidewalk_right_ : 1;
        uint32_t wheelchair_     : 1;
        uint32_t dismount_       : 1; // stairs, bike dismount

        // --- Pollu-specific attributes ---
        uint8_t aqi_penalty_;     // precomputed AQI penalty (0-255)
        uint8_t greenery_score_;  // canopy density 0-255
        uint8_t noise_level_;     // optional: environmental noise score 0-255
        uint8_t reserved_;

        OSMWay() { memset(this, 0, sizeof(OSMWay)); }

        OSMWay(uint64_t id) {
            memset(this, 0, sizeof(OSMWay));
            osmid_ = id;
        }

        void set_way_id(uint64_t id) { osmid_ = id; }
        uint64_t way_id() const { return osmid_; }

        void set_node_count(uint32_t c) { nodecount_ = c; }
        uint32_t node_count() const { return nodecount_; }

        void set_name_idx(uint32_t idx) { name_idx_ = idx; }
        bool has_name() const { return name_idx_ > 0; }

        void set_ref_idx(uint32_t idx) { ref_idx_ = idx; }
        bool has_ref() const { return ref_idx_ > 0; }
    };

    static_assert(sizeof(OSMWay) <= 32, "OSMWay grew; review memory layout");

} // namespace pollu


#endif //OSMWAY_H
