//
// Created by Shlok Jain on 16/08/25.
//

#ifndef PBFGRAPHPARSER_H
#define PBFGRAPHPARSER_H

#include "/Users/shlokjain/CLionProjects/pollu/src/nalanda/osmdata.h"  // Pollu's slim OSMData (file-backed)
#include <boost/property_tree/ptree.hpp>
#include <string>
#include <vector>

namespace pollu {
    namespace takshashila {
        /**
         * Parses OSM PBF extracts into file-backed temp stores for graph building.
         * Two-pass flow:
         *  1) ParseWays  -> writes {ways_file, way_nodes_file} and fills OSMData counters & name tables.
         *  2) ParseNodes -> writes {nodes_file} filtered to only node-ids referenced by ways.
         */
        class PBFGraphParser {
        public:
            /**
             * Pass 1: Parse ways and write file-backed outputs.
             *
             * @param pt                 Read-only config (Boost.PropertyTree).
             * @param input_files        List of .osm.pbf input paths.
             * @param ways_file          Output: contiguous POD OSMWay records.
             * @param way_nodes_file     Output: CSR layout (offsets + flat stream of OSM node-ids per way).
             *
             * @return OSMData           Staging container with counters, interning tables, maps, and file paths set.
             *                           (OSMData::ways_path / way_nodes_path are populated; nodes_path is empty here.)
             */
            static nalanda::OSMData ParseWays(const boost::property_tree::ptree& pt,
                                     const std::vector<std::string>& input_files,
                                     const std::string& ways_file,
                                     const std::string& way_nodes_file);

            /**
             * Pass 2: Parse nodes and write file-backed outputs (filtered to “needed” ids discovered in ParseWays).
             *
             * @param pt                 Read-only config (Boost.PropertyTree).
             * @param input_files        List of .osm.pbf input paths.
             * @param way_nodes_file     Output: contiguous references to nodes by Way IDs
             * @param nodes_file         Output: contiguous POD OSMNode records (fixed-point coords, flags).
             * @param osmdata            In/out: OSMData from ParseWays (provides needed-node set via internal state).
             *                           On success, osmdata.nodes_path is set and counters updated.
             */
            static void ParseNodes(const boost::property_tree::ptree& pt,
                                   const std::vector<std::string>& input_files,
                                   const std::string& way_nodes_file,
                                   const std::string& nodes_file,
                                   nalanda::OSMData& osmdata);

            // Non-instantiable utility class
            PBFGraphParser() = delete;
        };
    } // namespace takshashila

} // namespace pollu

#endif // PBFGRAPHPARSER_H
