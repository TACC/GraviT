/* 
 * File:   ThreadSafeQueue.h
 * Author: jbarbosa
 *
 * Created on May 31, 2014, 12:46 PM
 */

#ifndef GVT_CORE_SCHEDULE_THREAD_SAFE_QUEUE_H
#define	GVT_CORE_SCHEDULE_THREAD_SAFE_QUEUE_H


#include <boost/thread/mutex.hpp>
#include <boost/container/vector.hpp>


 namespace gvt {
    namespace core {
        namespace schedule {
        template<class T, bool threadsafe = true >
            class ThreadSafeQueue {
            public:


                typedef typename std::vector<T>::iterator iterator;
                typedef typename std::vector<T>::const_iterator const_iterator;

                ThreadSafeQueue() {

                }
                ThreadSafeQueue(const ThreadSafeQueue& orig) {
                    if(threadsafe) orig.lock();
                    _queue.assign(orig.begin(),orig.end());
                    if(threadsafe) orig.unlock();
                }

                operator std::vector<T>() {
                    return _queue;
                }

                virtual ~ThreadSafeQueue() {
                }

                inline bool pop(T& elem) {
                    if(threadsafe)boost::mutex::scoped_lock lock(_mqueue);
                    if (_queue.empty()) return false;
                    elem = *(_queue.begin());
                    _queue.erase(_queue.begin());
                    return true;
                }

                inline bool pop(std::vector<T>& list, size_t size = 1) {
                    boost::mutex::scoped_lock lock(_mqueue);
                    if (_queue.empty()) return false;
                    size_t qelem = _queue.size();
                    size_t nelem = std::min(size, _queue.size());
                    list.clear();
                    list.assign(_queue.begin(), _queue.begin() + nelem);
                    _queue.erase(_queue.begin(), _queue.begin() + nelem);
                    return true;
                }

                inline bool pop(ThreadSafeQueue& list, size_t size = 1) {
                    if(threadsafe)boost::mutex::scoped_lock lock(_mqueue);
                    if (_queue.empty()) return false;
                    size_t nelem = std::min(size, _queue.size());
                    list.clear();
                    list.assign(_queue.begin(), _queue.begin() + nelem);
                    _queue.erase(_queue.begin(), _queue.begin() + nelem);
                    return true;
                }

                inline bool push(T& elem) {
                    if(threadsafe)boost::mutex::scoped_lock lock(_mqueue);
                    _queue.push_back(elem);
                    return true;
                }

                inline bool push_back(T& elem) {
                    if(threadsafe)boost::mutex::scoped_lock lock(_mqueue);
                    _queue.push_back(elem);
                    return true;
                }

                inline bool push(std::vector<T>& list) {
                    if(threadsafe)boost::mutex::scoped_lock lock(_mqueue);
                    _queue.insert(_queue.end(), list.begin(), list.end());
                    return true;
                }

                inline bool push(ThreadSafeQueue& list) {
                    if(threadsafe)boost::mutex::scoped_lock lock(_mqueue);
                    _queue.insert(_queue.end(), list._queue.begin(), list._queue.end());
                    return true;
                }

                inline size_t size() {
                    return _queue.size();
                }

                inline bool empty() {
                    return _queue.empty();
                }

                inline void reserve(size_t size) {
                    if(threadsafe) boost::mutex::scoped_lock lock(_mqueue);
                    _queue.reserve(size);
                }

                inline iterator begin() {
                    return _queue.begin();
                }

                inline const_iterator begin() const {
                    return _queue.begin();
                }

                inline iterator end() {
                    return _queue.end();
                }

                inline const_iterator end() const {
                    return _queue.end();
                }


                inline iterator rbegin() {
                    return _queue.rbegin();
                }

                inline const_iterator rbegin() const {
                    return _queue.rbegin();
                }

                inline iterator rend() {
                    return _queue.rend();
                }

                inline const_iterator rend() const {
                    return _queue.rend();
                }

                inline const_iterator cbegin() const {
                    return _queue.cbegin();
                }

                inline const_iterator cend() const {
                    return _queue.cend();
                }

                inline void lock() {
                    _mqueue.lock();
                }

                inline void unlock() {
                    _mqueue.unlock();
                }

                inline T& back() {
                    return _queue.back();
                }

                inline T& front() {
                    return _queue.front();
                }

                inline void pop_back() {
                    if(threadsafe) boost::mutex::scoped_lock lock(_mqueue);
                    _queue.pop_back();
                }

                inline T& operator[](const int index) {
                    return _queue[index];
                }

                inline T& operator[](const int index) const {
                    return _queue[index];
                }

                inline void erase(const_iterator elem) {
                    if(threadsafe) boost::mutex::scoped_lock lock(_mqueue);
                    _queue.erase(elem);
                }
            template <class InputIterator>    
                inline void erase(InputIterator start, InputIterator end) {
                    if(threadsafe) boost::mutex::scoped_lock lock(_mqueue);
                    _queue.erase(start,end);
                }

            template <class InputIterator>
                inline void assign(InputIterator start, InputIterator end) {
                    if(threadsafe) boost::mutex::scoped_lock lock(_mqueue);
                    _queue.assign(start,end);
                }
            template <class InputIterator>
                inline void insert(const_iterator insertAt,InputIterator start,InputIterator end) {
                    if(threadsafe) boost::mutex::scoped_lock lock(_mqueue);
                    _queue.insert(insertAt,start,end);
                }

                inline void clear() {
                    if(threadsafe) boost::mutex::scoped_lock lock(_mqueue);
                    _queue.clear();
                }


                boost::container::vector<T> _queue;

            protected:
                boost::mutex _mqueue;
            };
        }
    }
}


#endif	/* GVT_CORE_SCHEDULE_THREAD_SAFE_QUEUE_H */

