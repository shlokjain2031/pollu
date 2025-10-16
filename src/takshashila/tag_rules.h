//
// Created by Shlok Jain on 16/10/25.
//

#ifndef TAG_RULES_H
#define TAG_RULES_H

#pragma once

#include <string>
#include <unordered_set>

struct TagRules {
    std::unordered_set<std::string> relevant_keys;
    std::unordered_set<std::string> unwalkable_values;
    std::unordered_set<std::string> walkable_highways;

    static TagRules LoadFromJSON(const std::string& path);
};


#endif //TAG_RULES_H
