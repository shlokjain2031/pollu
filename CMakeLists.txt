cmake_minimum_required(VERSION 3.30)
project(pollu)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(Protobuf REQUIRED)
find_package(ZLIB REQUIRED)
find_package(absl REQUIRED) # Abseil is now required for protobuf

# Optional (but recommended): detect libosmium via pkg-config
set(LIBOSMIUM_INCLUDE_DIR /opt/homebrew/include)

# Protozero is just headers, add manually if cloned locally
set(PROTOZERO_INCLUDE_DIR /Users/shlokjain/CLionProjects/pollu/third_party/protozero/include)

# System libs libosmium may rely on
find_package(Threads REQUIRED)
find_library(BZIP2_LIB bz2 REQUIRED)
find_library(EXPAT_LIB expat REQUIRED)

# Include dirs
include_directories(
        ${Protobuf_INCLUDE_DIRS}
        ${ZLIB_INCLUDE_DIRS}
        ${LIBOSMIUM_INCLUDE_DIRS}
        ${PROTOZERO_INCLUDE_DIR}  # uncomment if not in system include path
        third_party
)

set(SOURCES
        main.cpp
        third_party/fileformat.pb.cc
        third_party/osmformat.pb.cc
        src/takshashila/pbfgraphparser.h
        src/takshashila/pbfgraphparser.cpp
        src/takshashila/graphbuilder.cpp
        src/takshashila/graphbuilder.h
        src/nalanda/edge.h
        src/nalanda/node.h
        src/nalanda/osmnode.h
        src/nalanda/osmway.h
        src/nalanda/osmdata.h
        src/nalanda/uniquenames.h
        src/nalanda/disk_seq.h
)

add_executable(pollu ${SOURCES})

# Link everything together
target_link_libraries(pollu
        ${Protobuf_LIBRARIES}
        ${ZLIB_LIBRARIES}
        ${BZIP2_LIB}
        ${EXPAT_LIB}
        Threads::Threads
        absl::log_internal_check_op
        absl::log_internal_message
        absl::log_internal_nullguard
        absl::strings
        absl::base
)
