#include <iostream>

#include "src/takshashila/pbfgraphparser.h"

int main() {
    const std::string kPath = "/Users/shlokjain/CLionProjects/pollu/.osm.pbf/kansai-latest.osm.pbf";
    pollu::nalanda::OSMData osm;
    try {

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}
