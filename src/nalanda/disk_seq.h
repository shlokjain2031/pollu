//
// Created by Shlok Jain on 22/08/25.
//

#ifndef DISK_SEQ_H
#define DISK_SEQ_H
#pragma once

#include <type_traits>
#include <cstdint>
#include <string>

// Vendored Valhalla header:
#include "/Users/shlokjain/CLionProjects/pollu/third_party/valhalla_sequence.h"

namespace pollu {
    // Thin wrapper so the rest of Pollu never depends on Valhalla directly.
    // We also enforce a POD-like constraint and keep an obvious seam for future swap.
    template <class T>
    class DiskSeq : public valhalla::midgard::sequence<T> {
        static_assert(std::is_trivially_copyable_v<T>,
                      "pollu::storage::DiskSeq<T> requires T to be trivially copyable (no pointers/vtables/heap).");
        using Base = valhalla::midgard::sequence<T>;

    public:
        // Forward Valhalla constructors:
        // sequence(const std::string& file_name, bool create=false, size_t write_buffer_size=?)
        using Base::Base;

        // Placeholder durability hook (kept for future Pollu-native formats).
        // For now this is just a flush; later we can add fsync/headers without changing call sites.
        void commit() {
            this->flush();
        }
    };

    // Convenience helpers to make call sites explicit and tidy.
    template <class T>
    inline DiskSeq<T> open_seq(const std::string& path) {
        return DiskSeq<T>(path, /*create=*/false);
    }

    template <class T>
    inline DiskSeq<T> create_seq(const std::string& path, size_t write_buf_elems = (32u * 1024u * 1024u) / sizeof(T)) {
        return DiskSeq<T>(path, /*create=*/true, write_buf_elems);
    }

}


#endif //DISK_SEQ_H
