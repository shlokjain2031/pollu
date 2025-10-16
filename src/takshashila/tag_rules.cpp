//
// Created by Shlok Jain on 16/10/25.
//

#include "tag_rules.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <stdexcept>

using json = nlohmann::json;

TagRules TagRules::LoadFromJSON(const std::string& path) {
    TagRules rules;

    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open tag_rules.json: " + path);
    }

    json j;
    ifs >> j;

    if (!j.contains("tags") || !j["tags"].is_array()) {
        throw std::runtime_error("Invalid tag_rules.json format: missing 'tags' array.");
    }

    for (const auto& tag : j["tags"]) {
        if (!tag.contains("key") || !tag.contains("value")) continue;

        std::string key = tag["key"].get<std::string>();
        std::string value = tag["value"].get<std::string>();
        std::string desc = tag.value("description", "");

        // mark pedestrian-relevant keys
        if (key == "foot" || key == "pedestrian" ||
            key == "pedestrian:conditional" || key == "oneway:foot" ||
            key == "highway") {
            rules.relevant_keys.insert(key);
            }

        // detect unwalkable tags via description text
        std::string desc_lower = desc;
        std::transform(desc_lower.begin(), desc_lower.end(), desc_lower.begin(), ::tolower);
        if (desc_lower.find("not allowed") != std::string::npos ||
            desc_lower.find("routing not allowed") != std::string::npos) {
            rules.unwalkable_values.insert(value);
            }

        // collect walkable highway types
        if (key == "highway") {
            rules.walkable_highways.insert(value);
        }
    }

    return rules;
}
