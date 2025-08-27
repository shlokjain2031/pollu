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

        OSMWay() { memset(this, 0, sizeof(OSMWay)); }

        OSMWay(uint64_t id) {
            memset(this, 0, sizeof(OSMWay));
            osmid_ = id;
        }

        // --- Way ID ---
        void set_way_id(uint64_t id) { osmid_ = id; }
        uint64_t way_id() const { return osmid_; }

        // --- Node count ---
        void set_node_count(uint32_t c) { nodecount_ = static_cast<uint16_t>(c); }
        uint32_t node_count() const { return nodecount_; }

        // --- Classification ---
        void set_road_class(uint32_t c) { road_class_ = c; }
        uint32_t road_class() const { return road_class_; }

        void set_use(uint32_t u) { use_ = u; }
        uint32_t use() const { return use_; }

        void set_lanes(uint32_t l) { lanes_ = l; }
        uint32_t lanes() const { return lanes_; }

        void set_surface(uint32_t s) { surface_ = s; }
        uint32_t surface() const { return surface_; }

        // --- Routing flags ---
        void set_oneway(bool v) { oneway_ = v; }
        bool oneway() const { return oneway_; }

        void set_oneway_rev(bool v) { oneway_rev_ = v; }
        bool oneway_rev() const { return oneway_rev_; }

        void set_roundabout(bool v) { roundabout_ = v; }
        bool roundabout() const { return roundabout_; }

        void set_tunnel(bool v) { tunnel_ = v; }
        bool tunnel() const { return tunnel_; }

        void set_bridge(bool v) { bridge_ = v; }
        bool bridge() const { return bridge_; }

        void set_indoor(bool v) { indoor_ = v; }
        bool indoor() const { return indoor_; }

        void set_lit(bool v) { lit_ = v; }
        bool lit() const { return lit_; }

        void set_pedestrian_fwd(bool v) { pedestrian_fwd_ = v; }
        bool pedestrian_fwd() const { return pedestrian_fwd_; }

        void set_pedestrian_bwd(bool v) { pedestrian_bwd_ = v; }
        bool pedestrian_bwd() const { return pedestrian_bwd_; }

        void set_sidewalk_left(bool v) { sidewalk_left_ = v; }
        bool sidewalk_left() const { return sidewalk_left_; }

        void set_sidewalk_right(bool v) { sidewalk_right_ = v; }
        bool sidewalk_right() const { return sidewalk_right_; }

        void set_wheelchair(bool v) { wheelchair_ = v; }
        bool wheelchair() const { return wheelchair_; }

        void set_dismount(bool v) { dismount_ = v; }
        bool dismount() const { return dismount_; }
    };

    static_assert(sizeof(OSMWay) <= 32, "OSMWay grew; review memory layout");

} // namespace pollu


#endif //OSMWAY_H
