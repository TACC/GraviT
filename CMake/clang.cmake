message("Clang/LLVM compiler selected")

SET(GVT_ARCH_FLAGS__SSSE3 "-msse3")
SET(GVT_ARCH_FLAGS__SSSE3 "-mssse3")
SET(GVT_ARCH_FLAGS__SSE41 "-msse4.1")
SET(GVT_ARCH_FLAGS__SSE42 "-msse4.2")
SET(GVT_ARCH_FLAGS__AVX   "-mavx")
SET(GVT_ARCH_FLAGS__AVX2  "-mf16c -mavx2 -mfma -mlzcnt -mabm -mbmi -mbmi2 -fabi-version=6")

# SET(CMAKE_CXX_COMPILER "clang++")
# SET(CMAKE_C_COMPILER "clang")
# SET(CMAKE_CXX_FLAGS "-fPIC")
# SET(CMAKE_CXX_FLAGS_DEBUG          "--g -O3 -ftree-ter")
# SET(CMAKE_CXX_FLAGS_RELEASE        "-DNDEBUG    -O3 -no-ansi-alias -restrict -fp-model fast -fimf-precision=low -no-prec-div -no-prec-sqrt")
# SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-DNDEBUG -g -O3 -no-ansi-alias -restrict -fp-model fast -fimf-precision=low -no-prec-div -no-prec-sqrt")
# SET(CMAKE_EXE_LINKER_FLAGS "")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics -fPIC -Wno-expansion-to-defined -Wno-unknown-pragmas -Wno-writable-strings")
SET(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG}")
SET(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -O3")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3")
SET(CMAKE_EXE_LINKER_FLAGS "")

#-no-ansi-alias -restrict -fp-model fast -fimf-precision=low -no-prec-div -no-prec-sqrt"
