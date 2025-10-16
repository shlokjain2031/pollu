//
// Created by Shlok Jain on 16/08/25.
//

#include <boost/property_tree/ptree_fwd.hpp>

#include <osmium/io/any_input.hpp>
#include </Users/shlokjain/CLionProjects/pollu/src/nalanda/osmdata.h>
#include <valhalla_sequence.h>
#include <osmium/visitor.hpp>
#include <osmium/handler.hpp>
#include </Users/shlokjain/CLionProjects/pollu/src/nalanda/graph_constants.h>
#include <unordered_set>

#include "tag_rules.h"
#include "util.h"
#include "../nalanda/osmway.h"
#include "unordered_dense/include/ankerl/unordered_dense.h"

using Tags = ankerl::unordered_dense::map<std::string, std::string>;

struct graph_parser {

  graph_parser(pollu::nalanda::OSMData& osmdata)
    : osmdata_(osmdata) {

      // --- initialise state ---
      current_way_node_index_ = 0;
      last_node_ = 0;
      last_way_ = 0;
      average_speed_ = 0.0f;
      has_surface_tag_ = false;
      has_tracktype_tag_ = false;
      osmid_ = 0;

      // no Lua: just initialise empty tags
      empty_node_tags_ = Tags{};

      // --- initialise tag handlers ---
      tag_handlers_["duration"] = [this]() {
        std::size_t found = tag_.second.find(":");
        if (found != std::string::npos) {
          std::vector<std::string> time = GetTagTokens(tag_.second, ":");
          uint32_t hour = 0, min = 0, sec = 0;
          if (time.size() == 1) { // minutes
            std::stringstream ss(time.at(0));
            ss >> min;
            min *= 60;
          } else if (time.size() == 2) { // hours and min
            std::stringstream ss(tag_.second);
            ss >> hour;
            ss.ignore();
            hour *= 3600;

            ss >> min;
            min *= 60;
          } else if (time.size() == 3) { // hours, min, and sec
            std::stringstream ss(tag_.second);
            ss >> hour;
            ss.ignore();
            hour *= 3600;

            ss >> min;
            ss.ignore();
            min *= 60;

            ss >> sec;
          }
          way_.set_duration(hour + min + sec);
        }
      };

      tag_handlers_["indoor"] = [this]() {
        way_.set_indoor(tag_.second == "true" ? true : false); ;
      };

      tag_handlers_["tunnel"] = [this]() {
        way_.set_tunnel(tag_.second == "true" ? true : false); ;
      };

      tag_handlers_["bridge"] = [this]() {
        way_.set_bridge(tag_.second == "true" ? true : false); ;
      };

      tag_handlers_["pedestrian_forward"] = [this]() {
        way_.set_pedestrian_fwd(tag_.second == "true" ? true : false); ;
      };

      tag_handlers_["pedestrian_backward"] = [this]() {
        way_.set_pedestrian_bwd(tag_.second == "true" ? true : false);
      };

      tag_handlers_["oneway"] = [this]() {
        way_.set_oneway(tag_.second == "true" ? true : false); ;
      };

      tag_handlers_["oneway_reverse"] = [this]() {
        way_.set_oneway_rev(tag_.second == "true" ? true : false); ;
      };
  }

  struct Way {
    uint64_t osmid;
    std::vector<uint64_t> nodes;
    Tags tags;
    uint64_t changeset_id;
  };

static void transform_way(const osmium::Way& way,
                          const Tags& empty_way_tags,
                          std::vector<Way>& transformed) {
  // --- 1. Collect node references ---
  std::vector<uint64_t> nodes;
  nodes.reserve(way.nodes().size());
  for (const auto& node : way.nodes()) {
    nodes.push_back(node.ref());
  }

  // Skip degenerate ways (<2 nodes)
  if (nodes.size() < 2) {
    return;
  }

  // --- 2. Skip closed polygons representing non-routable areas ---
  if (nodes.front() == nodes.back()) {
    for (const auto& tag : way.tags()) {
      std::string_view key = tag.key();
      if (key == "building" || key == "landuse" ||
          key == "leisure" || key == "natural") {
        return; // skip closed non-routable area
      }
    }
  }

  // --- 3. Load tag rules (once) ---
  static TagRules pedestrian_rules =
      TagRules::LoadFromJSON("/Users/shlokjain/CLionProjects/pollu/tag_rules.json");

  bool walkable = false;
  bool explicitly_unwalkable = false;
  bool has_relevant_tag = false;

  Tags tags; // transformed tags to retain

  const osmium::TagList& map_tags = way.tags();
  for (const auto& tag : map_tags) {
    const char* key_c = tag.key();
    const char* value_c = tag.value();
    if (!key_c || !value_c) continue;

    std::string key = to_lower(std::string(key_c));
    std::string value = to_lower(std::string(value_c));

    // --- A. Keep only relevant tags (defined in JSON) ---
    if (pedestrian_rules.relevant_keys.count(key)) {
      has_relevant_tag = true;
      tags.emplace(key, value);
    }

    // --- B. Check for explicit unwalkable values (from JSON) ---
    if (pedestrian_rules.unwalkable_values.count(value)) {
      explicitly_unwalkable = true;
      break;
    }

    // --- C. Access logic for pedestrian/foot tags ---
    if (key == "foot" || key == "pedestrian") {
      if (pedestrian_rules.unwalkable_values.count(value)) {
        explicitly_unwalkable = true;
        break;
      }
      if (value == "yes" || value == "allowed" || value == "public" ||
          value == "permissive" || value == "designated" ||
          value == "official" || value == "sidewalk" || value == "footway" ||
          value == "passable") {
        walkable = true;
      }
      else if (value == "private" || value == "permit" || value == "residents") {
        explicitly_unwalkable = true;
        break;
      }
    }

    // --- D. Conditional restrictions ---
    else if (key.find(":conditional") != std::string::npos) {
      if (value.rfind("no", 0) == 0) {  // starts with "no"
        explicitly_unwalkable = true;
        break;
      } else {
        walkable = true;
      }
    }

    // --- E. Highway-based defaults (from JSON) ---
    else if (key == "highway") {
      if (pedestrian_rules.walkable_highways.count(value)) {
        walkable = true;
      }
    }
  }

  // --- 4. Filtering decisions ---
  if (!has_relevant_tag) {
    return; // no relevant tags → skip
  }
  if (explicitly_unwalkable || !walkable) {
    return; // unsuitable for pedestrian routing
  }
  if (tags.empty()) {
    return;
  }

  transformed.emplace_back(
      Way{
        static_cast<uint64_t>(way.id()),
        std::move(nodes),
        std::move(tags),
        way.changeset()
      });
}


  using TagHandler = std::function<void()>;
  std::unordered_map<std::string, TagHandler> tag_handlers_;
  pollu::nalanda::OSMWay way_;
  std::pair<std::string, std::string> tag_;
  uint64_t osmid_;
  float average_speed_ = 0.0f;
  bool has_surface_ = true;
  bool has_surface_tag_ = true;
  bool has_tracktype_tag_ = true;
  pollu::nalanda::OSMData& osmdata_;
  std::unique_ptr<valhalla::midgard::sequence<pollu::nalanda::OSMWay>> ways_;
  std::unique_ptr<valhalla::midgard::sequence<pollu::nalanda::OSMWayNode>> way_nodes_;
  size_t current_way_node_index_;
  uint64_t last_node_;
  uint64_t last_way_;
  ankerl::unordered_dense::map<uint64_t, size_t> loop_nodes_;
  Tags empty_node_tags_;
};

namespace pollu {
    namespace takshashila {
        static void ParseNodes(const boost::property_tree::ptree& pt,
                                   const std::vector<std::string>& input_files,
                                   const std::string& way_nodes_file,
                                   const std::string& nodes_file,
                                   nalanda::OSMData& osmdata) {

        }
    }
} // namespace pollu
