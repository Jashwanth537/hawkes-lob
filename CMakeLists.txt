cmake_minimum_required(VERSION 3.15)
project(lob_collector CXX)

# Build (macOS):
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release -DOPENSSL_ROOT_DIR=$(brew --prefix openssl)
#   make -j$(nproc)
#
# Build (Ubuntu — system packages already installed):
#   mkdir build && cd build && cmake .. && make -j2

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── nlohmann/json ─────────────────────────────────────────────────────────────
# Use system package when available (Ubuntu: nlohmann-json3-dev), otherwise
# fetch the header-only release tarball.
find_package(nlohmann_json 3.2.0 QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    )
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    set(JSON_Install   OFF CACHE INTERNAL "")
    FetchContent_MakeAvailable(json)
    message(STATUS "nlohmann/json: using FetchContent")
else()
    message(STATUS "nlohmann/json: using system package")
endif()

# ── IXWebSocket ───────────────────────────────────────────────────────────────
# If $HOME/IXWebSocket contains a CMakeLists.txt (pre-cloned on the VM), wire
# it in directly as a subdirectory — avoids a redundant git clone.
# Otherwise fall back to FetchContent (macOS dev machine, CI, etc.).
set(USE_TLS             ON  CACHE BOOL "" FORCE)
set(IXWEBSOCKET_INSTALL OFF CACHE BOOL "" FORCE)

set(_IXWS_LOCAL "$ENV{HOME}/IXWebSocket")
if(EXISTS "${_IXWS_LOCAL}/CMakeLists.txt")
    message(STATUS "IXWebSocket: using local source at ${_IXWS_LOCAL}")
    add_subdirectory("${_IXWS_LOCAL}"
                     "${CMAKE_CURRENT_BINARY_DIR}/ixwebsocket-build"
                     EXCLUDE_FROM_ALL)
else()
    message(STATUS "IXWebSocket: fetching from GitHub")
    include(FetchContent)
    FetchContent_Declare(
        ixwebsocket
        GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
        GIT_TAG        v11.4.5
    )
    FetchContent_MakeAvailable(ixwebsocket)
endif()

# ── OpenSSL (explicit link in case IXWebSocket doesn't re-export it) ──────────
find_package(OpenSSL REQUIRED)

# ── collector binary ──────────────────────────────────────────────────────────
add_executable(collector main.cpp)

target_link_libraries(collector PRIVATE
    nlohmann_json::nlohmann_json
    ixwebsocket
    OpenSSL::SSL
    OpenSSL::Crypto
    $<$<NOT:$<PLATFORM_ID:Windows>>:pthread>
)

target_compile_options(collector PRIVATE
    -O2 -Wall -Wextra -Wno-unused-parameter
)
