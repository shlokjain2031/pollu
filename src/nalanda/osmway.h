//
// Created by Shlok Jain on 21/08/25.
//

#ifndef OSMWAY_H
#define OSMWAY_H

#pragma once
#include <cstdint>
#include <cstring>

namespace pollu {
    namespace nalanda {
        struct OSMWay {
            uint64_t osmid_;   // OSM way id
            uint16_t nodecount_;  // number of nodes in way

            // --- Classification ---
            uint32_t highway_class : 3;  // trunk, primary, residential, footway...
            uint8_t footway_        : 6;  // footway, steps, alley, cycleway, etc.

            // --- Routing flags ---
            uint32_t oneway_     : 1;
            uint32_t oneway_rev_ : 1;
            uint32_t tunnel_     : 1;
            uint32_t bridge_     : 1;
            uint32_t indoor_     : 1;

            // --- Travel hints ---
            uint32_t duration_;   // optional duration in seconds (ferries/hiking)

            uint32_t pedestrian_fwd_ : 1;
            uint32_t pedestrian_bwd_ : 1;

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
            void set_road_class(uint32_t c) { highway_class = c; }
            uint32_t road_class() const { return highway_class; }

            // --- Routing flags ---
            void set_oneway(bool v) { oneway_ = v; }
            bool oneway() const { return oneway_; }

            void set_oneway_rev(bool v) { oneway_rev_ = v; }
            bool oneway_rev() const { return oneway_rev_; }

            void set_tunnel(bool v) { tunnel_ = v; }
            bool tunnel() const { return tunnel_; }

            void set_bridge(bool v) { bridge_ = v; }
            bool bridge() const { return bridge_; }

            void set_indoor(bool v) { indoor_ = v; }
            bool indoor() const { return indoor_; }

            void set_duration(uint32_t d) { duration_ = d; }
            uint32_t duration() const { return duration_; }

            void set_pedestrian_fwd(bool v) { pedestrian_fwd_ = v; }
            bool pedestrian_fwd() const { return pedestrian_fwd_; }

            void set_pedestrian_bwd(bool v) { pedestrian_bwd_ = v; }
            bool pedestrian_bwd() const { return pedestrian_bwd_; }
        };

        static_assert(sizeof(OSMWay) <= 32, "OSMWay grew; review memory layout");
    } // namespace nalanda
} // namespace pollu


#endif //OSMWAY_H
