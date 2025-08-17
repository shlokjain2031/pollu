//
// Created by Shlok Jain on 16/08/25.
//

#ifndef PBF_READER_H
#define PBF_READER_H

#pragma once
#include "/Users/shlokjain/CLionProjects/pollu/src/nalanda/osm_structures.h"
#include <string>

namespace takshashila {
    class PBFLoader {
    public:
        static void load_from_file(const std::string& filepath, OSMData& out);
    };
}


#endif //PBF_READER_H
