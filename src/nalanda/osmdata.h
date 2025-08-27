//
// Created by Shlok Jain on 21/08/25.
//

#ifndef OSMDATA_H
#define OSMDATA_H
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include "UniqueNames.h"   // your interning table
// (OSMNode/OSMWay *definitions* are still used for POD record I/O, but not stored here as vectors)

namespace pollu {
  namespace nalanda {
    // way_id -> name index (forward / reverse)
    using OSMStringMap = std::unordered_map<uint64_t, uint32_t>;

    struct OSMData {
      // -------- lifecycle --------
      bool write_to_temp_files(const std::string& dir);   // write interning tables & small maps
      bool read_from_temp_files(const std::string& dir);  // read them back
      static void cleanup_temp_files(const std::string& dir);

      // -------- counts & sizing (kept verbatim) --------
      uint64_t max_changeset_id_ = 0;    // newest changeset id seen during parse
      uint64_t osm_node_count = 0;       // raw OSM nodes seen
      uint64_t osm_way_count = 0;        // raw OSM ways seen
      uint64_t osm_way_node_count = 0;   // total OSM node refs across all ways

      uint64_t node_count = 0;           // compact nodes written to nodes_file (after filtering)
      uint64_t edge_count = 0;           // estimated edges for prealloc in graph build

      // -------- relation-driven name updates (kept; harmless) --------
      OSMStringMap way_ref;              // way_id -> name index (forward)
      OSMStringMap way_ref_rev;          // way_id -> name index (reverse)

      // -------- string interning --------
      UniqueNames node_names;            // node-only strings
      UniqueNames name_offset_map;       // road names, refs, turn-lane strings, etc.

      // -------- file-backed paths --------
      std::string ways_path;             // contiguous OSMWay records
      std::string way_nodes_path;        // CSR node-id stream for each way
      std::string nodes_path;            // contiguous OSMNode records (filtered set)

      // Optional: store “needed node ids” spill path if you can’t keep it in RAM
      std::string needed_node_ids_path;  // e.g., sorted uint64_t list or bitset/bloom

      bool initialized = false;
    };
  } // namespace nalanda

} // namespace pollu


#endif //OSMDATA_H
