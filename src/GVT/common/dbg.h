/* 
 * File:   debug.h
 * Author: jbarbosa
 *
 * Created on March 10, 2014, 8:46 PM
 */

#ifndef DEBUG_H
#define	DEBUG_H

#include <cstdio>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>
#include <sstream>
#include <iostream>
#define __FILE_SHORT__ (strstr(__FILE__, "src/") ? strstr(__FILE__, "src/") + 4 : __FILE__)

const char * const DBG_COLOR_NORMAL = "\033[0m";
const char * const DBG_COLOR_RED = "\033[1;31m";
const char * const DBG_COLOR_GREEN = "\033[1;32m";
const char * const DBG_COLOR_YELLOW = "\033[1;33m";
const char * const DBG_COLOR_BLUE = "\033[1;34m";
const char * const DBG_COLOR_GRAY = DBG_COLOR_NORMAL;//"\033[1;37m";

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

#ifndef NDEBUG



enum DEBUG_LEVEL {
    DBG_NONE,
    DBG_ALWAYS,
    DBG_SEVERE,
    DBG_MODERATE,
    DBG_LOW
};

#define DEBUG_LEVEL DBG_ALWAYS

#define DEBUG_RANK false
#define DEBUG(x)
#define SUDO_DEBUG(x) 


template<typename T>
inline std::string to_string(T value) {
    std::string s;
    std::stringstream out;
    out << value;
    s = out.str();
    return s;
}

#define GVT_ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << DBG_COLOR_RED << "Failed assertion `" << DBG_COLOR_BLUE <<  #condition << DBG_COLOR_RED << " [" << DBG_COLOR_NORMAL  << __FILE_SHORT__ \
                      << " : " << __LINE__  << DBG_COLOR_RED << "]: " <<  DBG_COLOR_GRAY << message << DBG_COLOR_NORMAL << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (false)

#define GVT_ASSERT_BACKTRACE(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << DBG_COLOR_RED << "Failed assertion `" << DBG_COLOR_BLUE <<  #condition << DBG_COLOR_RED << " [" << DBG_COLOR_NORMAL  << __FILE_SHORT__ \
                      << " : " << __LINE__  << DBG_COLOR_RED << "]: " <<  DBG_COLOR_GRAY << message << DBG_COLOR_NORMAL << std::endl; \
            print_trace();\
            std::exit(EXIT_FAILURE); \
        } \
    } while (false)


#define GVT_DEBUG(level,message) \
    do { \
        if ( (level <= DEBUG_LEVEL)) { \
            std::cerr << DBG_COLOR_GREEN << "Debug[" << DBG_COLOR_NORMAL << __FILE_SHORT__ \
                      << ":" << __LINE__ << DBG_COLOR_GREEN << "]: " << DBG_COLOR_GRAY << message << DBG_COLOR_NORMAL << std::endl; \
        } \
    } while (false)

#define GVT_WARNING(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << DBG_COLOR_YELLOW << "Warning `" << DBG_COLOR_BLUE << #condition << DBG_COLOR_YELLOW << "` failed [" << DBG_COLOR_NORMAL  << __FILE_SHORT__ \
                      << ":" << __LINE__ << DBG_COLOR_YELLOW << "]: " << DBG_COLOR_GRAY << message << DBG_COLOR_NORMAL << std::endl; \
        } \
    } while (false)

#define GVT_WARNING_BACKTRACE(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << DBG_COLOR_YELLOW << "Warning `" << DBG_COLOR_BLUE << #condition << DBG_COLOR_YELLOW << "` failed [" << DBG_COLOR_NORMAL  << __FILE_SHORT__ \
                      << ":" << __LINE__ << DBG_COLOR_YELLOW << "]: " << DBG_COLOR_GRAY << message << DBG_COLOR_NORMAL << std::endl; \
            print_trace();\
        } \
    } while (false)

#else
#define GVT_WARNING(condition, message)
#define GVT_WARNING_BACKTRACE(condition, message)
#define GVT_DEBUG(level,message)
#define GVT_ASSERT(condition, message) \
    do { \
        if (! (condition) ) { \
            std::cerr << DBG_COLOR_RED << "ERROR:`" << DBG_COLOR_BLUE <<  #condition << DBG_COLOR_RED << ":" <<  DBG_COLOR_GRAY << message << DBG_COLOR_NORMAL << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (false)

#define GVT_ASSERT_BACKTRACE(condition, message) \
    do { \
        if (! (condition) ) { \
            std::cerr << DBG_COLOR_RED << "ERROR: `" << DBG_COLOR_BLUE <<  #condition << DBG_COLOR_RED << " [" << DBG_COLOR_NORMAL  << __FILE_SHORT__ \
                      << " : " << __LINE__  << DBG_COLOR_RED << "]: " <<  DBG_COLOR_GRAY << message << DBG_COLOR_NORMAL << std::endl; \
            print_trace();\
            std::exit(EXIT_FAILURE); \
        } \
    } while (false)

#define DEBUG_RANK false
#define DEBUG(x)
#define SUDO_DEBUG(x)
#endif




#endif	/* DEBUG_H */

