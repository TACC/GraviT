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
#ifndef MISC_TIMER
#define MISC_TIMER

#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

#include <mpi.h>

namespace gvt {
namespace core {
namespace time {

#if GVT_USE_TIMING

typedef std::chrono::high_resolution_clock clock_type;

/**
 * \brief Timer wrapper class for code profiling
 */
struct timer {

  std::chrono::time_point<clock_type> t_start; /**< Last time start/resume was called*/
  std::chrono::time_point<clock_type> t_end;   /**< Last time stop was called*/
  double total_elapsed;                        /**< Total time in ms accumulated by the timer */
  bool running;                                /**< Is the timer running */
  std::string text;                            /**< Timer name/text for results io */

  /**
   * Constructor
   * @method timer
   * @param  running Should the timer start running or not
   * @param  text    Text string for io
   */
  inline timer(bool running = true, std::string text = "") : text(text), total_elapsed(0), running(running) {
    if (running) {
      t_end = clock_type::now();
      t_start = clock_type::now();
    }
  }

  /**
   * Copy constructor
   * @method timer
   * @param  other Timer to copy
   */
  inline timer(const timer &other) {
    text = other.text;
    total_elapsed = other.total_elapsed;
  }

  /**
   * Assign operator, copies elsaped time
   */
  inline void operator=(const timer &other) {
    // text = other.text;
    total_elapsed = other.total_elapsed;
  }

  /**
   * Set time text for io
   * @method settext
   * @param  str     Text
   */
  inline void settext(std::string str) { text = str; }
  /**
   * \brief Destructor
   *
   * If the io text is not empty, prints the elapsed value
   *
   */
  inline ~timer() {
    auto end = clock_type::now();
    if (MPI::COMM_WORLD.Get_rank() == 0 && !text.empty()) print();
  }

  /**
   * If the timer is not running already starts timer
   * @method start
   */
  inline void start() {
    if (!running) {
      t_start = clock_type::now();
      t_end = clock_type::now();
      running = true;
    }
  }

  /**
   * Stop timer and accumulates elapsed time
   * @method stop
   */
  inline void stop() {
    if (running) {
      t_end = clock_type::now();
      total_elapsed += std::chrono::duration<double, std::milli>(t_end - t_start).count();
      running = false;
    }
  }

  /**
   * Restarts the timer
   * @method resume
   * @see start
   */
  inline void resume() {
    if (!running) {
      start();
    }
  }

  /**
   * Returns a formated string with the timer text and elapsed time. If the timer is running stops timer before
   * returning the elapsed time.
   * @method format
   * @return [description]
   */
  inline std::string format() const {
    double elapsed = total_elapsed;
    if (running) elapsed += std::chrono::duration<double, std::milli>(clock_type::now() - t_start).count();

    std::ostringstream os;
    os << elapsed << " ms";
    return os.str();
  }

  /**
   * Print to stdout the current elapsed time
   * @method print
   */
  inline void print() { std::cout << text << " " << format() << std::endl; }

  /**
   * \brief Adds two timers elapsed time.
   */
  inline timer operator+=(timer &other) {
    timer ret(false);
    ret.total_elapsed += other.total_elapsed;
    return ret;
  }

  /**
   * \brief Calculates timer elapsed difference.
   */

  inline timer operator-=(timer &other) {
    timer ret(false);
    ret.total_elapsed -= other.total_elapsed;
    return ret;
  }

  /**
   * \brief Add two elapsed time
   */
  friend inline timer operator+(const timer &a, const timer &b) {
    timer ret(false);
    ret.total_elapsed = a.total_elapsed + b.total_elapsed;
    return ret;
  }
  /**
   * \brief Calculates the difference of two timers elapsed time.
   */
  friend inline timer operator-(const timer &a, const timer &b) {
    timer ret(false);
    ret.total_elapsed = a.total_elapsed - b.total_elapsed;
    return ret;
  }

  friend std::ostream &operator<<(std::ostream &os, const timer &t) { return os << t.text << " :" << t.format(); }
};
#else
struct timer {
  /**
   * Constructor
   * @method timer
   * @param  running Should the timer start running or not
   * @param  text    Text string for io
   */
  timer(bool running = true, std::string text = "") {}
  /**
   * \brief Destructor
   *
   * If the io text is not empty, prints the elapsed value
   *
   */
  ~timer() {}
  /**
   * Set time text for io
   * @method settext
   * @param  str     Text
   */
  inline void settext(std::string str) {}
  /**
   * If the timer is not running already starts timer
   * @method start
   */
  inline void start() {}
  /**
   * Stop timer and accumulates elapsed time
   * @method stop
   */
  inline void stop() {}
  /**
   * Restarts the timer
   * @method resume
   * @see start
   */
  inline void resume() {}
  /**
   * Returns a formated string with the timer text and elapsed time. If the timer is running stops timer before
   * returning the elapsed time.
   * @method format
   * @return [description]
   */
  inline std::string format() const { return std::string(); }
  /**
   * \brief Adds two timers elapsed time.
   */
  inline timer operator+=(timer &other) { return timer(); }
  /**
   * \brief Calculates timer elapsed difference.
   */

  inline timer operator-=(timer &other) { return timer(); }
  /**
   * \brief Add two elapsed time
   */
  friend inline timer operator+(const timer &a, const timer &b) { return timer(); }
  /**
   * \brief Calculates the difference of two timers elapsed time.
   */
  friend inline timer operator-(const timer &a, const timer &b) { return timer(); }

  friend std::ostream &operator<<(std::ostream &os, const timer &t) { return os; }
};
#endif
}
}
}

#endif
