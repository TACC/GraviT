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
/*
 * File:   TaskScheduling.h
 * Author: jbarbosa
 *
 * Created on May 28, 2014, 10:53 PM
 */

#ifndef GVT_CORE_SCHEDULE_TASK_SCHEDULING_H
#define GVT_CORE_SCHEDULE_TASK_SCHEDULING_H
#include <iostream>
#include <queue>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/atomic.hpp>
#include <boost/container/vector.hpp>
#include <boost/foreach.hpp>

namespace gvt {
namespace core {
namespace schedule {

typedef boost::function<void()> Task;

/// base class for threaded GraviT components
/** base class for asynchronously-executing components of GraviT.
This class manages thread execution and synchronization
*/
class asyncExec {
protected:
  std::queue<boost::function<void()> > tasks_;
  boost::thread_group threads_;

  boost::atomic<std::size_t> wcounter;
  boost::mutex mutex_;
  boost::mutex msingleton_;
  boost::condition_variable condition_;
  bool running_;

public:
  std::size_t numThreads;
  static asyncExec *_sinstance;

  asyncExec(std::size_t numThreads = boost::thread::hardware_concurrency() / 2)
      : numThreads(numThreads), running_(true), wcounter(0) {
    for (std::size_t i = 0; i < numThreads; ++i) {
      threads_.create_thread(boost::bind(&asyncExec::pool_main, this));
    }
  }

  ~asyncExec() {
    {
      boost::unique_lock<boost::mutex> lock(mutex_);
      running_ = false;
      condition_.notify_all();
    }
    // clang-format off
    try {
      threads_.join_all();
    }
    catch (...) {
    }
    // clang-format on
  }

  /**
   * Retrieve an instance to the thread pool singleton.
   */
  static asyncExec *instance() {
    if (!_sinstance) _sinstance = new asyncExec();
    return _sinstance;
  }

  /**
   * Add a task to the thread pool to be executed.
   */
  template <typename T> void run_task(T task) {
    boost::unique_lock<boost::mutex> lock(mutex_);
    wcounter++;
    tasks_.push(task);
    condition_.notify_one();
  }

  /**
   * Block until all pending tasks are complete.
   */
  void sync() {
    while (!tasks_.empty() || wcounter > 0) {
      asm("");
    }
  }

private:
  void pool_main() {
    while (running_) {
      boost::unique_lock<boost::mutex> lock(mutex_);
      while (tasks_.empty() && running_) {
        condition_.wait(lock);
      }
      if (!running_) break;
      {
        boost::function<void()> task = tasks_.front();
        tasks_.pop();
        lock.unlock();
        // clang-format off
        try {
          task();
        }
        catch (...) {
        }
        // clang-format on
        wcounter--;
      }
    }
  }
};
}
}
}

#endif /* GVT_CORE_SCHEDULE_TASK_SCHEDULING_H */
