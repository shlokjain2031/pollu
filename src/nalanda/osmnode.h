//
// Created by Shlok Jain on 21/08/25.
//

#ifndef OSMNODE_H
#define OSMNODE_H

#include <cstdint>
#include <cstring>
#include <limits>
#include <cmath>

namespace pollu {

constexpr uint32_t kMaxNodeNameIndex = 2097151; // 21-bit max

struct OSMNode {
  // OSM node id (64-bit from raw data)
  uint64_t osmid_;

  // ---- Routing-relevant flags ----
  uint64_t traffic_signal_ : 1;
  uint64_t stop_sign_      : 1;
  uint64_t minor_          : 1;
  uint64_t direction_      : 1;
  uint64_t spare_bits_     : 16; // reserved for Pollu-specific flags later

  // ---- Node type & intersection ----
  uint32_t type_          : 4;  // compact NodeType enum
  uint32_t is_intersection_: 1; // junction after compaction
  uint32_t spare_meta_    : 27; // reserved

  // ---- Coordinates (1e7 fixed-point) ----
  uint32_t lon_e7_;
  uint32_t lat_e7_;

  // ---- Constructors ----
  OSMNode() { memset(this, 0, sizeof(OSMNode)); }

  OSMNode(const uint64_t id, double lat = std::numeric_limits<double>::max(),
          double lon = std::numeric_limits<double>::max()) {
    memset(this, 0, sizeof(OSMNode));
    set_id(id);
    set_latlon(lat, lon);
  }

  // ---- ID ----
  void set_id(const uint64_t id) { osmid_ = id; }
  uint64_t id() const { return osmid_; }

  // ---- Lat/Lon setters & getters ----
  void set_latlon(double lat, double lon) {
    auto L = std::llround((lon + 180.0) * 1e7);
    lon_e7_ = (L >= 0 && L <= static_cast<long long>(360 * 1e7)) ?
              static_cast<uint32_t>(L) : std::numeric_limits<uint32_t>::max();

    auto B = std::llround((lat + 90.0) * 1e7);
    lat_e7_ = (B >= 0 && B <= static_cast<long long>(180 * 1e7)) ?
              static_cast<uint32_t>(B) : std::numeric_limits<uint32_t>::max();
  }

  double lon() const {
    return lon_e7_ == std::numeric_limits<uint32_t>::max() ?
           std::numeric_limits<double>::quiet_NaN() :
           (lon_e7_ * 1e-7 - 180.0);
  }

  double lat() const {
    return lat_e7_ == std::numeric_limits<uint32_t>::max() ?
           std::numeric_limits<double>::quiet_NaN() :
           (lat_e7_ * 1e-7 - 90.0);
  }

  bool valid_coords() const {
    return lon_e7_ != std::numeric_limits<uint32_t>::max() &&
           lat_e7_ != std::numeric_limits<uint32_t>::max();
  }

  // ---- Index setters & guards ----
  // ---- Routing flags ----
  void set_traffic_signal(bool v) { traffic_signal_ = v; }
  bool traffic_signal() const { return traffic_signal_; }

  void set_stop_sign(bool v) { stop_sign_ = v; }
  bool stop_sign() const { return stop_sign_; }

  void set_minor(bool v) { minor_ = v; }
  bool minor() const { return minor_; }

  void set_direction(bool v) { direction_ = v; }
  bool direction() const { return direction_; }

  void set_intersection(bool v) { is_intersection_ = v; }
  bool intersection() const { return is_intersection_; }

  // ---- Node type ----
  enum class NodeType : uint8_t { Unknown=0, Regular=1, Barrier=2, TransitStop=3, FerryTerminal=4 };
  void set_type(NodeType t) { type_ = static_cast<uint32_t>(t); }
  NodeType type() const { return static_cast<NodeType>(type_); }
};

static_assert(sizeof(OSMNode) <= 48, "OSMNode grew; review memory layout");

} // namespace pollu

#endif //OSMNODE_H
