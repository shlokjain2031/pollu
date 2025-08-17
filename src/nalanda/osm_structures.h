//
// Created by Shlok Jain on 16/08/25.
//

#ifndef OSM_STRUCTURES_H
#define OSM_STRUCTURES_H

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

struct OSMNode {
    using id_t = std::int64_t;
    id_t id = 0;
    double lat = 0.0, lon = 0.0;
    std::unordered_map<std::string, std::string> tags;
};

struct OSMWay {
    using id_t = std::int64_t;
    id_t id = 0;
    std::vector<OSMNode::id_t> node_refs;
    std::unordered_map<std::string, std::string> tags;
};

struct OSMRelation {
    using id_t = std::int64_t;
    enum class MemberType { Node, Way, Relation };
    struct Member { MemberType type; std::int64_t ref; std::string role; };

    id_t id = 0;
    std::vector<Member> members;
    std::unordered_map<std::string, std::string> tags;
};

struct OSMData {
    std::vector<OSMNode> nodes;
    std::vector<OSMWay> ways;
    std::vector<OSMRelation> relations;

    void clear() { nodes.clear(); ways.clear(); relations.clear(); }
};


#endif //OSM_STRUCTURES_H
