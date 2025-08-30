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
      tag_handlers_["use"] = [this]() {
        Use use = static_cast<Use>(std::stoi(tag_.second));
          switch (use) {
            case Use::kCycleway:
              way_.set_use(Use::kCycleway);
              break;
            case Use::kFootway:
              way_.set_use(Use::kFootway);
              break;
            case Use::kSidewalk:
              way_.set_use(Use::kSidewalk);
              break;
            case Use::kPedestrian:
              way_.set_use(Use::kPedestrian);
              break;
            case Use::kPath:
              way_.set_use(Use::kPath);
              break;
            case Use::kElevator:
              way_.set_use(Use::kElevator);
              break;
            case Use::kSteps:
              way_.set_use(Use::kSteps);
              break;
            case Use::kEscalator:
              way_.set_use(Use::kEscalator);
              break;
            case Use::kBridleway:
              way_.set_use(Use::kBridleway);
              break;
            case Use::kPedestrianCrossing:
              way_.set_use(Use::kPedestrianCrossing);
              break;
            case Use::kLivingStreet:
              way_.set_use(Use::kLivingStreet);
              break;
            case Use::kAlley:
              way_.set_use(Use::kAlley);
              break;
            case Use::kEmergencyAccess:
              way_.set_use(Use::kEmergencyAccess);
              break;
            case Use::kServiceRoad:
              way_.set_use(Use::kServiceRoad);
              break;
            case Use::kTrack:
              way_.set_use(Use::kTrack);
              break;
            case Use::kOther:
              way_.set_use(Use::kOther);
              break;
            case Use::kConstruction:
              way_.set_use(Use::kConstruction);
              break;
            case Use::kRoad:
            default:
              way_.set_use(Use::kRoad);
              break;
          }
      };

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

      tag_handlers_["surface"] = [this]() {
          std::string value = tag_.second;
          std::transform(value.begin(), value.end(), value.begin(),
                     [](unsigned char c){ return std::tolower(c); });

          // Find unpaved before paved since they have common string
          if (value.find("unpaved") != std::string::npos) {
            way_.set_surface(Surface::kGravel);

          } else if (value.find("paved") != std::string::npos ||
                     value.find("pavement") != std::string::npos ||
                     value.find("asphalt") != std::string::npos ||
                     // concrete, concrete:lanes, concrete:plates
                     value.find("concrete") != std::string::npos ||
                     value.find("cement") != std::string::npos ||
                     value.find("chipseal") != std::string::npos ||
                     value.find("metal") != std::string::npos) {
            way_.set_surface(Surface::kPavedSmooth);

          } else if (value.find("tartan") != std::string::npos ||
                     value.find("pavingstone") != std::string::npos ||
                     value.find("paving_stones") != std::string::npos ||
                     value.find("sett") != std::string::npos ||
                     value.find("grass_paver") != std::string::npos) {
            way_.set_surface(Surface::kPaved);

          } else if (value.find("cobblestone") != std::string::npos ||
                     value.find("brick") != std::string::npos) {
            way_.set_surface(Surface::kPavedRough);

          } else if (value.find("compacted") != std::string::npos ||
                     value.find("wood") != std::string::npos ||
                     value.find("boardwalk") != std::string::npos) {
            way_.set_surface(Surface::kCompacted);

          } else if (value.find("dirt") != std::string::npos ||
                     value.find("natural") != std::string::npos ||
                     value.find("earth") != std::string::npos ||
                     value.find("ground") != std::string::npos ||
                     value.find("mud") != std::string::npos) {
            way_.set_surface(Surface::kDirt);

          } else if (value.find("gravel") != std::string::npos || // gravel, fine_gravel
                     value.find("pebblestone") != std::string::npos ||
                     value.find("sand") != std::string::npos) {
            way_.set_surface(Surface::kGravel);
          } else if (value.find("grass") != std::string::npos ||
                     value.find("stepping_stones") != std::string::npos) {
            way_.set_surface(Surface::kPath);
            // We have to set a flag as surface may come before Road classes and Uses
          } else {
            has_surface_ = false;
          }
        };

      tag_handlers_["roundabout"] = [this]() {
        way_.set_roundabout(tag_.second == "true" ? true : false); ;
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

      tag_handlers_["sidewalk"] = [this]() {
        if (tag_.second == "both" || tag_.second == "yes" || tag_.second == "shared" ||
            tag_.second == "raised") {
          way_.set_sidewalk_left(true);
          way_.set_sidewalk_right(true);
          } else if (tag_.second == "left") {
            way_.set_sidewalk_left(true);
          } else if (tag_.second == "right") {
            way_.set_sidewalk_right(true);
          }
      };

      tag_handlers_["lit"] = [this]() {
        way_.set_lit(tag_.second == "true" ? true : false); ;
      };

      tag_handlers_["oneway"] = [this]() {
        way_.set_oneway(tag_.second == "true" ? true : false); ;
      };

      tag_handlers_["oneway_reverse"] = [this]() {
        way_.set_oneway_rev(tag_.second == "true" ? true : false); ;
      };
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
