/* 
 * File:   AdaptiveSend.h
 * Author: jbarbosa
 *
 * Created on January 21, 2014, 4:18 PM
 */

#ifndef ADAPTIVESEND_H
#define	ADAPTIVESEND_H

#include "HybridBaseSchedule.h"

struct AdaptiveSendSchedule : public HybridBaseSchedule {
    

    AdaptiveSendSchedule(int * newMap, int &size, int *map_size_buf, int **map_recv_bufs, int *data_send_buf) : HybridBaseSchedule(newMap, size, map_size_buf, map_recv_bufs,data_send_buf) {

    }

    virtual ~AdaptiveSendSchedule() {

    }

    virtual void operator()() {
        DEBUG(cerr << "in AdaptiveSend scheduler" << endl);
        for (int i = 0; i < size; ++i)
            newMap[i] = -1;

        std::map< int, std::vector<int> > cur_data2procs;
        std::map< int, std::vector<int> > send_data;
        std::map< int, int > data2size;
        std::map< int, int > procs_count;
        for (int s = 0; s < size; ++s) {
            if (map_recv_bufs[s]) {
                // add currently loaded data
                // track number of procs that have data loaded
                cur_data2procs[map_recv_bufs[s][0]].push_back(s);
                procs_count[map_recv_bufs[s][0]] += 1;
                DEBUG(cerr << "    noting currently " << s << " -> " << map_recv_bufs[s][0] << endl);

                // add data with queued rays
                // track number of rays needed (assumes map inits to zero)
                for (int d = 1; d < map_size_buf[s]; d += 2) {
                    data2size[map_recv_bufs[s][d]] += map_recv_bufs[s][d + 1];
                    DEBUG(cerr << "        " << s << " has " << map_recv_bufs[s][d + 1] << " rays for data " << map_recv_bufs[s][d] << endl);
                }
            }
        }

        // put data into priority buckets
        // based on number of rays waiting for data
        std::map< int, std::vector< int > > priority;
        std::map< int, std::vector< int > > zero_priority;
        //int add_a_proc = atts.GetNewProcThreshold();
        int add_a_proc = 5000; // XXX TODO: hacked!
        int avail = 0;
        for (std::map<int, int>::iterator it = data2size.begin(); it != data2size.end(); ++it) {
            int idx = it->second / add_a_proc;
            DEBUG(cerr << "    data " << it->first << " has priority " << idx << " (" << it->second << " rays)" << endl);
            if (idx > 0) {
                priority[idx].push_back(it->first);
            } else {
                std::map<int, int>::iterator pdit = procs_count.find(it->first);
                if (pdit != procs_count.end()) {
                    for (int i = 0; i < pdit->second; ++i) {
                        DEBUG(cerr << "            adding " << cur_data2procs[pdit->first][i] << " to zero-priority pool at " << i << endl);
                        zero_priority[i].push_back(cur_data2procs[pdit->first][i]); // push proc id onto stack, let map sort into use-first order
                    }
                    avail += pdit->second;
                    DEBUG(cerr << "        adding " << pdit->second << " zero-priority procs to available pool, is now " << avail << endl);
                }
            }
        }

        // track procs that have data with no pending rays
        // give them artificially high sort order so they'll be used first
        if (!priority.empty()) {
            for (std::map<int, std::vector<int> >::iterator it = cur_data2procs.begin(); it != cur_data2procs.end(); ++it) {
                if (data2size.find(it->first) == data2size.end()) {
                    int pc = procs_count[it->first];
                    for (int i = 0; i < pc; ++i) {
                        DEBUG(cerr << "            adding " << it->second[i] << " to zero-priority pool at " << ((pc << 6) + i) << endl);
                        zero_priority[(pc << 6) + i].push_back(it->second[i]);
                    }
                    avail += pc;
                    DEBUG(cerr << "        adding " << pc << " rayless procs to available pool, is now " << avail << endl);
                }
            }
        }

        DEBUG(cerr << "    prioritization done." << endl;
                cerr << "        " << priority.size() << " priority levels" << endl;
                cerr << "        " << zero_priority.size() << " zero-priority levels" << endl;
                cerr << "        " << avail << " processors available" << endl;
                );

        // while procs are available
        // get highest priority and allocate procs
        // std::vector<int> excess;  // reclaim excess procs? or reclaim them when become zero priority?
        DEBUG(cerr << "    scheduling" << endl);
        for (std::map< int, std::vector<int> >::reverse_iterator rit = priority.rbegin(); (rit != priority.rend()) & (avail > 0); ++rit) {
            for (std::vector<int>::iterator it = rit->second.begin(); (it != rit->second.end()) & (avail > 0); ++it) {
                int need = rit->first - procs_count[*it];
                DEBUG(cerr << "    data " << *it << " needs " << need << " procs of " << avail << " available" << endl);
                if (procs_count[*it] > 0) {
                    DEBUG(cerr << "        already has " << procs_count[*it] << " allocated" << endl);
                    // add already-allocated procs
                    int used;
                    for (used = 0; (used < procs_count[*it]) & (need > 0); ++used) {
                        int p = cur_data2procs[*it][used];
                        newMap[p] = *it;
                        --need;
                        DEBUG(cerr << "        adding " << p << " -> " << *it << " to map" << endl);
                    }

                    // put remaining processor into empty queue,
                    // with high sort order so they get used first
                    if (used < procs_count[*it]) {
                        int rem = (procs_count[*it] - used) << 10;
                        for (int i = used; i < procs_count[*it]; ++i) {
                            zero_priority[rem + i].push_back(cur_data2procs[*it].back());
                            cur_data2procs[*it].pop_back();
                            ++avail;
                        }
                        DEBUG(cerr << "        returned " << (rem >> 10) << " procs to avail pool, is now " << avail << endl);
                    } else {
                        need = (need > avail) ? avail : (need > 0) ? need : 0;
                        avail -= need;
                        DEBUG(cerr << "        taking " << need << " additional procs" << endl);
                        // data already loaded, put rest in send queue
                        std::map<int, std::vector<int> >::reverse_iterator zpi = zero_priority.rbegin();
                        for (int i = 0; i < need; ++i) {
                            int victim = zpi->second.back();
                            newMap[victim] = *it;
                            send_data[*it].push_back(victim);
                            zpi->second.pop_back();
                            if (zpi->second.empty()) {
                                ++zpi;
                            }
                            DEBUG(cerr << "        adding " << victim << " -> " << *it << " to map" << endl);
                        }
                        if (zpi != zero_priority.rbegin()) {
                            --zpi;
                            int target = zpi->first;
                            DEBUG(cerr << "        erasing zero_priority nodes from " << target << " to end" << endl);
                            zero_priority.erase(zero_priority.find(target), zero_priority.end());
                        }
                    }
                } else {
                    // not loaded anywhere.  Load on one, put rest in to_send queue
                    // mark to load on one
                    need = (need > avail) ? avail : (need > 0) ? need : 0;
                    avail -= need;

                    DEBUG(cerr << "        not loaded anywhere, taking " << need << " procs" << endl);
                    std::map<int, std::vector<int> >::reverse_iterator zpi = zero_priority.rbegin();
                    int victim = zpi->second.back();
                    newMap[victim] = *it;
                    cur_data2procs[*it].push_back(victim); // this proc will load the data, then send to the rest
                    zpi->second.pop_back();
                    if (zpi->second.empty()) {
                        ++zpi;
                    }
                    DEBUG(cerr << "        adding " << victim << " -> " << *it << " to map" << endl);

                    // put rest in send queue
                    // start from 1, to account for first just added
                    for (int i = 1; i < need; ++i) {
                        victim = zpi->second.back();
                        newMap[victim] = *it;
                        send_data[*it].push_back(victim);
                        zpi->second.pop_back();
                        if (zpi->second.empty()) {
                            ++zpi;
                        }
                        DEBUG(cerr << "        adding " << victim << " -> " << *it << " to map" << endl);
                    }
                    if (zpi != zero_priority.rbegin()) {
                        --zpi;
                        int target = zpi->first;
                        DEBUG(cerr << "        erasing zero_priority nodes from " << target << " to end" << endl);
                        zero_priority.erase(zero_priority.find(target), zero_priority.end());
                    }
                }
            }
        }

        DEBUG(cerr << "    priority allocation done, " << avail << " procs still available" << endl);

        // allocate zero-priority if that's all that's present
        // use adaptive allocation scheme
        if (priority.empty()) {
            DEBUG(cerr << "    doing zero-priority allocation" << endl);

            std::map<int, int> size2data;
            // convert data2size into size2data,
            // use data id to pseudo-uniqueify, since only need ordering
            for (std::map<int, int>::iterator it = data2size.begin(); it != data2size.end(); ++it) {
                size2data[(it->second << 7) + it->first] = it->first;
            }

            // iterate over queued data, find which are already loaded somewhere
            // if there are multiple procs with one data, just keep one
            std::vector<int> homeless;
            for (std::map<int, int>::iterator d2sit = size2data.begin(); d2sit != size2data.end(); ++d2sit) {
                std::map< int, std::vector<int> >::iterator d2pit = cur_data2procs.find(d2sit->second);
                if (d2pit != cur_data2procs.end()) {
                    newMap[d2pit->second[0]] = d2pit->first;
                    DEBUG(cerr << "    adding " << d2pit->second[0] << " -> " << d2pit->first << " to map" << endl);
                } else {
                    homeless.push_back(d2sit->second);
                    DEBUG(cerr << "    noting " << d2sit->second << " is homeless" << endl);
                }
            }

            // iterate over newMap, fill as many procs as possible with homeless data
            // could be dupes in the homeless list, so keep track of what's added
            for (int i = 0; (i < size) & (!homeless.empty()); ++i) {
                if (newMap[i] < 0) {
                    while (!homeless.empty()
                            && cur_data2procs.find(homeless.back()) != cur_data2procs.end())
                        homeless.pop_back();
                    if (!homeless.empty()) {
                        newMap[i] = homeless.back();
                        cur_data2procs[newMap[i]].push_back(i);
                        homeless.pop_back();
                        DEBUG(cerr << "    adding " << i << " -> " << newMap[i] << " to map" << endl);
                    }
                }
            }
        }

        // build data send buffer
        // index is proc to receive, value is proc to send
        DEBUG(cerr << "building data set buffer" << endl);
        for (std::map<int, std::vector<int> >::iterator it = send_data.begin(); it != send_data.end(); ++it) {
            int to_recv = it->second.size();
            int to_send = cur_data2procs[it->first].size();
            DEBUG(cerr << "    data " << it->first << " needed on " << to_recv << " procs, already on " << to_send << " procs" << endl);
            for (int r = 0, s = 0; r < to_recv; ++r, ++s) {
                s = r % to_send; // XXX TODO pnav: 's' is likely an error here, but confirm and fix
                DEBUG(cerr << "    for data " << it->first << ": " << it->second[r] << " <- " << cur_data2procs[it->first][s] << endl);
                data_send_buf[it->second[r]] = cur_data2procs[it->first][s];
            }
        }

        DEBUG(cerr << "new map size is " << size << endl;
        for (int i = 0; i < size; ++i)
                cerr << "    " << i << " -> " << newMap[i] << endl;
                );
        DEBUG(cerr << "data_send buffer is" << endl;
        for (int i = 0; i < size; ++i)
            if (data_send_buf[i] < 0)
                    cerr << "    " << i << " keeps" << endl;
            else
                cerr << "    " << i << " <-- " << data_send_buf[i] << endl;
                    );
    }

};


#endif	/* ADAPTIVESEND_H */

