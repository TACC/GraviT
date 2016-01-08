/* =======================================================================================
   This file is released as part of GraviT - scalable, platform independent ray tracing
   tacc.github.io/GraviT

   Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas at Austin
   All rights reserved.

   Licensed under the BSD 3-Clause License, (the "License"); you may not use this file
   except in compliance with the License.
   A copy of the License is included with this software in the file LICENSE.
   If your copy does not contain the License, you may obtain a copy of the License at:

       http://opensource.org/licenses/BSD-3-Clause

   Unless required by applicable law or agreed to in writing, software distributed under
   the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.
   See the License for the specific language governing permissions and limitations under
   limitations under the License.

   GraviT is funded in part by the US National Science Foundation under awards ACI-1339863,
   ACI-1339881 and ACI-1339840
   ======================================================================================= */
#ifndef GVT_CORE_DEBUG_H
#define GVT_CORE_DEBUG_H

#include <cstdio>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>
#include <sstream>
#include <iostream>
#include <cstring>
#define __FILE_SHORT__ (strstr(__FILE__, "src/") ? std::strstr(__FILE__, "src/") + 4 : __FILE__)

const char *const DBG_COLOR_NORMAL = "\033[0m";
const char *const DBG_COLOR_RED = "\033[1;31m";
const char *const DBG_COLOR_GREEN = "\033[1;32m";
const char *const DBG_COLOR_YELLOW = "\033[1;33m";
const char *const DBG_COLOR_BLUE = "\033[1;34m";
const char *const DBG_COLOR_GRAY = DBG_COLOR_NORMAL; //"\033[1;37m";

inline void print_trace() {
  char pid_buf[30];
  sprintf(pid_buf, "%d", getpid());
  char name_buf[512];
  name_buf[readlink("/proc/self/exe", name_buf, 511)] = 0;
  int child_pid = fork();
  if (!child_pid) {
    dup2(2, 1); // redirect output to stderr
    std::cerr << "stack trace for " << name_buf << " pid= " << pid_buf << std::endl;
#ifndef __APPLE__
    execlp("gdb", "gdb", "--batch", "-n", "-ex", "thread", "-ex", "bt", name_buf, pid_buf, NULL);
#else
    execlp("gdb-apple", "gdb-apple", "--batch", "-n", "-ex", "thread", "-ex", "bt", name_buf, pid_buf, NULL);
#endif
    abort(); /* If gdb failed to start */
  } else {
    waitpid(child_pid, NULL, 0);
  }
}

/// define GVT_USE_DEBUG to enable inline debugging messages
#ifdef GVT_USE_DEBUG

using std::cout;
using std::cerr;
using std::endl;
using std::flush;

/// GraviT debug level.
/** GraviT debug levels used to grade debug output.
If the debug statement level is at or above the set debug level, the statement will print.
*/
enum GVT_DEBUG_LEVEL {
  DBG_NONE, // should never be used in debug statements, set to this to turn off all debugging output
  DBG_ALWAYS,
  DBG_SEVERE,
  DBG_MODERATE,
  DBG_LOW,
  DBG_OFF // used to keep a debug statement in place, but to turn it off, without excessive comments
};

/// GraviT debug level, see GVT_DEBUG_LEVEL enum
#define DEBUG_LEVEL DBG_ALWAYS

// XXX TODO - remove these from source
#define DEBUG_RANK 0
#define DEBUG(x)
#define SUDO_DEBUG(x)
// end XXX TODO - remove these from source

template <typename T> inline std::string to_string(T value) {
  std::string s;
  std::stringstream out;
  out << value;
  s = out.str();
  return s;
}

#define GVT_ASSERT(condition, message)                                                                                 \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      std::cerr << DBG_COLOR_RED << "Failed assertion `" << DBG_COLOR_BLUE << #condition << DBG_COLOR_RED << " ["      \
                << DBG_COLOR_NORMAL << __FILE_SHORT__ << " : " << __LINE__ << DBG_COLOR_RED << "]: " << DBG_COLOR_GRAY \
                << message << DBG_COLOR_NORMAL << std::endl;                                                           \
      std::exit(EXIT_FAILURE);                                                                                         \
    }                                                                                                                  \
  } while (false)

#define GVT_ASSERT_BACKTRACE(condition, message)                                                                       \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      std::cerr << DBG_COLOR_RED << "Failed assertion `" << DBG_COLOR_BLUE << #condition << DBG_COLOR_RED << " ["      \
                << DBG_COLOR_NORMAL << __FILE_SHORT__ << " : " << __LINE__ << DBG_COLOR_RED << "]: " << DBG_COLOR_GRAY \
                << message << DBG_COLOR_NORMAL << std::endl;                                                           \
      print_trace();                                                                                                   \
      std::exit(EXIT_FAILURE);                                                                                         \
    }                                                                                                                  \
  } while (false)

#define GVT_DEBUG(level, message)                                                                                      \
  do {                                                                                                                 \
    if ((level <= DEBUG_LEVEL)) {                                                                                      \
      std::cerr << DBG_COLOR_GREEN << "Debug[" << DBG_COLOR_NORMAL << __FILE_SHORT__ << ":" << __LINE__                \
                << DBG_COLOR_GREEN << "]: " << DBG_COLOR_GRAY << message << DBG_COLOR_NORMAL << std::endl;             \
    }                                                                                                                  \
  } while (false)

#define GVT_WARNING(condition, message)                                                                                \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      std::cerr << DBG_COLOR_YELLOW << "Warning `" << DBG_COLOR_BLUE << #condition << DBG_COLOR_YELLOW << "` failed [" \
                << DBG_COLOR_NORMAL << __FILE_SHORT__ << ":" << __LINE__ << DBG_COLOR_YELLOW                           \
                << "]: " << DBG_COLOR_GRAY << message << DBG_COLOR_NORMAL << std::endl;                                \
    }                                                                                                                  \
  } while (false)

#define GVT_WARNING_BACKTRACE(condition, message)                                                                      \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      std::cerr << DBG_COLOR_YELLOW << "Warning `" << DBG_COLOR_BLUE << #condition << DBG_COLOR_YELLOW << "` failed [" \
                << DBG_COLOR_NORMAL << __FILE_SHORT__ << ":" << __LINE__ << DBG_COLOR_YELLOW                           \
                << "]: " << DBG_COLOR_GRAY << message << DBG_COLOR_NORMAL << std::endl;                                \
      print_trace();                                                                                                   \
    }                                                                                                                  \
  } while (false)

#define GVT_DEBUG_CODE(level, block)                                                                                   \
  do {                                                                                                                 \
    if ((level <= DEBUG_LEVEL)) {                                                                                      \
      block;                                                                                                           \
    }                                                                                                                  \
  } while (false)

#else // !defined GVT_USE_DEBUG

#define GVT_WARNING(condition, message)
#define GVT_WARNING_BACKTRACE(condition, message)
#define GVT_DEBUG(level, message)
#define GVT_ASSERT(condition, message)                                                                                 \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      std::cerr << DBG_COLOR_RED << "ERROR:`" << DBG_COLOR_BLUE << #condition << DBG_COLOR_RED << ":"                  \
                << DBG_COLOR_GRAY << message << DBG_COLOR_NORMAL << std::endl;                                         \
      std::exit(EXIT_FAILURE);                                                                                         \
    }                                                                                                                  \
  } while (false)

#define GVT_ASSERT_BACKTRACE(condition, message)                                                                       \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      std::cerr << DBG_COLOR_RED << "ERROR: `" << DBG_COLOR_BLUE << #condition << DBG_COLOR_RED << " ["                \
                << DBG_COLOR_NORMAL << __FILE_SHORT__ << " : " << __LINE__ << DBG_COLOR_RED << "]: " << DBG_COLOR_GRAY \
                << message << DBG_COLOR_NORMAL << std::endl;                                                           \
      print_trace();                                                                                                   \
      std::exit(EXIT_FAILURE);                                                                                         \
    }                                                                                                                  \
  } while (false)

#define GVT_DEBUG_CODE(level, block)

// XXX TODO - remove these from source
#define DEBUG_RANK false
#define DEBUG(x)
#define SUDO_DEBUG(x)
// end XXX TODO - remove these from source

#endif // defined GVT_USE_DEBUG

#endif // GVT_CORE_DEBUG_H
