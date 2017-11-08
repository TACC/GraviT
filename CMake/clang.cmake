message("Clang/LLVM compiler selected")

SET(GVT_ARCH_FLAGS__SSSE3 "-msse3")
SET(GVT_ARCH_FLAGS__SSSE3 "-mssse3")
SET(GVT_ARCH_FLAGS__SSE41 "-msse4.1")
SET(GVT_ARCH_FLAGS__SSE42 "-msse4.2")
SET(GVT_ARCH_FLAGS__AVX   "-mavx")
SET(GVT_ARCH_FLAGS__AVX2  "-mf16c -mavx2 -mfma -mlzcnt  -mbmi -mbmi2")
SET(GVT_ARCH_FLAGS__AVX512KNL "-march=knl")
SET(GVT_ARCH_FLAGS__AVX512SKX "-march=skx")

# SET(CMAKE_CXX_COMPILER "clang++")
# SET(CMAKE_C_COMPILER "clang")
# SET(CMAKE_CXX_FLAGS "-fPIC")
# SET(CMAKE_CXX_FLAGS_DEBUG          "--g -O3 -ftree-ter")
# SET(CMAKE_CXX_FLAGS_RELEASE        "-DNDEBUG    -O3 -no-ansi-alias -restrict -fp-model fast -fimf-precision=low -no-prec-div -no-prec-sqrt")
# SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-DNDEBUG -g -O3 -no-ansi-alias -restrict -fp-model fast -fimf-precision=low -no-prec-div -no-prec-sqrt")
# SET(CMAKE_EXE_LINKER_FLAGS "")

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fcolor-diagnostics")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics -fPIC -Wno-c++11-narrowing -Wno-unknown-pragmas -Wno-writable-strings")
SET(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} -g")
SET(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -O3 -flto")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -g -O3 -flto")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}-flto")

#-no-ansi-alias -restrict -fp-model fast -fimf-precision=low -no-prec-div -no-prec-sqrt"
