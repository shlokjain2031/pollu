#include "src/nalanda/osm_structures.h"
#include <iostream>

#include "src/takshashila/pbf_reader.h"

int main() {
    const std::string kPath = "/Users/shlokjain/CLionProjects/pollu/.osm.pbf/kansai-latest.osm.pbf";
    OSMData osm;
    try {
        takshashila::PBFLoader::load_from_file(kPath, osm);
        std::cout << "Nodes: " << osm.nodes.size() << "\n"
                  << "Ways: " << osm.ways.size() << "\n"
                  << "Relations: " << osm.relations.size() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}
