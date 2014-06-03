/* 
 * File:   TaskScheduling.h
 * Author: jbarbosa
 *
 * Created on May 28, 2014, 10:53 PM
 */

#ifndef TASKSCHEDULING_H
#define	TASKSCHEDULING_H
#include <iostream>
#include <queue>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/atomic.hpp>
#include <boost/container/vector.hpp>
#include <boost/foreach.hpp>


namespace GVT {
    namespace Concurrency {

        typedef boost::function< void() > Task;

        class asyncExec {
        protected:
            std::queue< boost::function< void() > > tasks_;
            boost::thread_group threads_;
            
            boost::atomic<std::size_t> wcounter;
            boost::mutex mutex_;
            boost::mutex msingleton_;
            boost::condition_variable condition_;
            bool running_;

            
            //static asyncTaskExecution _singleton;

        public:
            std::size_t numThreads;
            static asyncExec* _sinstance;
            
            asyncExec(std::size_t numThreads = boost::thread::hardware_concurrency() / 2 )
            : numThreads(numThreads), running_(true), wcounter(0) {
                for (std::size_t i = 0; i < numThreads; ++i) {
                    threads_.create_thread(boost::bind(&asyncExec::pool_main, this));
                }
            }

            ~asyncExec() {
                {
                    boost::unique_lock< boost::mutex > lock(mutex_);
                    running_ = false;
                    condition_.notify_all();
                }

                try {
                    threads_.join_all();
                } catch (...) {
                }
            }

            static asyncExec* instance() {
                if(!_sinstance) _sinstance = new asyncExec();
                return _sinstance;
            }

            template < typename T >
            void run_task(T task) {
                boost::unique_lock< boost::mutex > lock(mutex_);
                wcounter++;
                tasks_.push(task);
                condition_.notify_one();
            }
            template < typename T >
            void sync(T task) {
            }

            void sync() {
                while (!tasks_.empty() || wcounter > 0) {
                    asm("");
                }
            }

        private:

            void pool_main() {
                while (running_) {
                    boost::unique_lock< boost::mutex > lock(mutex_);
                    while (tasks_.empty() && running_) {
                        condition_.wait(lock);
                    }
                    if (!running_) break;
                    {
                        boost::function< void() > task = tasks_.front();
                        tasks_.pop();
                        lock.unlock();
                        try {
                            task();
                        } catch (...) {
                        }
                        wcounter--;
                    }
                }
            }
        };
    }
}



#endif	/* TASKSCHEDULING_H */

