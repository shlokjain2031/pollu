//
// Created by Shlok Jain on 16/08/25.
//

#include "pbf_reader.h"
#include "/Users/shlokjain/CLionProjects/pollu/src/nalanda/osm_structures.h"
#include <osmium/io/any_input.hpp>
#include <osmium/visitor.hpp>
#include <osmium/handler.hpp>

namespace takshashila {

    class CollectorHandler : public osmium::handler::Handler {
    public:
        explicit CollectorHandler(OSMData& d) : data(d) {}
        void node(const osmium::Node& n);
        void way(const osmium::Way& w);
        void relation(const osmium::Relation& r);
    private:
        OSMData& data;
    };

    void CollectorHandler::node(const osmium::Node& n) {
        OSMNode out;
        out.id = n.id();
        if (n.location().valid()) { out.lat = n.location().lat(); out.lon = n.location().lon(); }
        for (auto& t : n.tags()) out.tags[t.key()] = t.value();
        data.nodes.emplace_back(std::move(out));
    }

    void CollectorHandler::way(const osmium::Way& w) {
        OSMWay out;
        out.id = w.id();
        for (auto& nref : w.nodes()) out.node_refs.push_back(nref.ref());
        for (auto& t : w.tags()) out.tags[t.key()] = t.value();
        data.ways.emplace_back(std::move(out));
    }

    void CollectorHandler::relation(const osmium::Relation& r) {
        OSMRelation out;
        out.id = r.id();
        for (auto& m : r.members()) {
            OSMRelation::Member mem;
            mem.ref = m.ref();
            mem.role = m.role() ? m.role() : "";
            mem.type = (m.type() == osmium::item_type::node) ? OSMRelation::MemberType::Node :
                      (m.type() == osmium::item_type::way) ? OSMRelation::MemberType::Way :
                                                             OSMRelation::MemberType::Relation;
            out.members.push_back(std::move(mem));
        }
        for (auto& t : r.tags()) out.tags[t.key()] = t.value();
        data.relations.emplace_back(std::move(out));
    }

    void PBFLoader::load_from_file(const std::string& filepath, OSMData& out) {
        osmium::io::Reader reader{filepath};
        CollectorHandler handler{out};
        osmium::apply(reader, handler);
        reader.close();
    }

} // namespace takshashila
