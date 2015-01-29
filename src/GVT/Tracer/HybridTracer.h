/* 
 * File:   HybridTracer.h
 * Author: jbarbosa
 *
 * Created on January 21, 2014, 10:46 AM
 */

#ifndef GVT_HYBRIDTRACER_H
#define	GVT_HYBRIDTRACER_H
#include <mpi.h>

#include <GVT/MPI/mpi_wrappers.h>
#include <GVT/Tracer/Tracer.h>
#include <GVT/Backend/MetaProcessQueue.h>
#include <GVT/Scheduler/schedulers.h>
#include <GVT/Concurrency/TaskScheduling.h>
#include <boost/foreach.hpp>


namespace GVT {

    namespace Trace {
        /// Tracer Hybrid (HybridSchedule) based decomposition implementation

                /*
         */
        template<class DomainType, class MPIW, class SCHEDULER> class Tracer<DomainType, MPIW, HybridSchedule<SCHEDULER> > : public Tracer_base<MPIW> {
        public:

            GVT::Domain::Domain* dom;
            int domTarget;
            long domain_counter;

            Tracer(GVT::Data::RayVector& rays, Image& image) : Tracer_base<MPIW>(rays, image) {
                int ray_portion = this->rays.size() / this->world_size;
                this->rays_start = this->rank * ray_portion;
                this->rays_end = (this->rank + 1) == this->world_size ? this->rays.size() : (this->rank + 1) * ray_portion; // tack on any odd rays to last proc

            }

            ~Tracer() {
            };

            virtual void operator()() {

                DEBUG(if (DEBUG_RANK) cerr << this->rank << ": " << (this->rays_end - this->rays_start) << " rays" << endl);

                long ray_counter = 0;

                DEBUG(cerr << "generating camera rays" << endl);

                this->generateRays();

                DEBUG(cerr << "tracing rays" << endl);


                // process domains until all rays are terminated
                bool all_done = false;
                std::map<int, int> doms_to_send;
                int domTargetCount = 0, lastDomain = -1;
                GVT::Domain::Domain* dom_mailbox = NULL;
                GVT::Data::RayVector moved_rays;
                moved_rays.reserve(1000);

                // begin by process domain with most rays queued
                for (map<int, GVT::Data::RayVector>::iterator q = this->queue.begin(); q != this->queue.end(); ++q) {
                    if (q->second.size() > domTargetCount) {
                        domTargetCount = q->second.size();
                        domTarget = q->first;
                    }
                }
                DEBUG(cerr << "selected domain " << domTarget << " (" << domTargetCount << " rays)" << endl);
                SUDO_DEBUG(if (DEBUG_RANK) cerr << "selected domain " << domTarget << " (" << domTargetCount << " rays)" << endl);

                while (!all_done) {

                    schedule(this->queue, domTarget, lastDomain, dom_mailbox, moved_rays, ray_counter);
                    // done with current domain, send off rays to their proper processors.

                    SendRays();

                    // are we done?

                    // root proc takes empty flag from all procs
                    int not_done = (int) (!this->queue.empty());
                    int *empties = (this->rank == 0) ? new int[this->world_size] : NULL;
                    MPI_Gather(&not_done, 1, MPI_INT, empties, 1, MPI_INT, 0, MPI_COMM_WORLD);

                    if (this->rank == 0) {
                        not_done = 0;
                        for (int i = 0; i < this->world_size; ++i)
                            not_done += empties[i];
                        for (int i = 0; i < this->world_size; ++i)
                            empties[i] = not_done;
                    }

                    MPI_Scatter(empties, 1, MPI_INT, &not_done, 1, MPI_INT, 0, MPI_COMM_WORLD);

                    DEBUG(if (DEBUG_RANK) cerr << this->rank << ": " << not_done << " procs still have rays" << " (my q:" << this->queue.size() << ")" << endl);
                    DEBUG(if (DEBUG_RANK)
                        for (map<int, GVT::Data::RayVector>::iterator q = this->queue.begin(); q != this->queue.end(); ++q)
                                cerr << "    q(" << q->first << "):" << q->second.size() << endl
                                );

                    all_done = (not_done == 0);

                    delete[] empties;

                }

                this->gatherFramebuffers(this->rays.size());

            }

            virtual bool schedule(std::map<int, GVT::Data::RayVector> &queue, int &domTarget, int &lastDomain, GVT::Domain::Domain* dom_mailbox, GVT::Data::RayVector &moved_rays, long &ray_counter) {
                dom = NULL;
                if (!this->queue.empty()) {
                    // pnav: use this to ignore domain x:        int domi=0;if (0)
                    if (domTarget >= 0) {
                        DEBUG(cerr << "Getting domain " << domTarget << endl);
                        SUDO_DEBUG(if (DEBUG_RANK) cerr << this->rank << ": Getting domain " << domTarget << endl);

                        if (domTarget != lastDomain)
                            if (dom != NULL) dom->free();

                        // register the dataset so it doesn't get deleted
                        // if we build a lightmap
                        if (dom_mailbox != NULL) {
                            dom = dom_mailbox;
                            dom_mailbox = NULL;
                            // was registered when it was loaded
                            DEBUG(if (DEBUG_RANK) cerr << "using dataset mailbox" << endl);
                            // don't increment counter here, since it wasn't loaded from disk
                            if (domTarget != lastDomain) {
                                lastDomain = domTarget;
                            }
                        } else {
                            dom = GVT::Env::RayTracerAttributes::rta->dataset->getDomain(domTarget);
                            SUDO_DEBUG(if (DEBUG_RANK) cerr << this->rank << ": called GetDomain for dataset: " << dom << endl);
                            // track domain loads
                            if (domTarget != lastDomain) {
                                ++domain_counter;
                                lastDomain = domTarget;
                                dom->load();
                            }
                        }

                        //GVT::Backend::ProcessQueue<DomainType>(new GVT::Backend::adapt_param<DomainType>(this->queue, moved_rays, domTarget, dom, this->colorBuf, ray_counter, domain_counter))();
                        {
                            moved_rays.reserve(this->queue[domTarget].size()*10);
                            boost::timer::auto_cpu_timer t("Tracing domain rays %t\n");
                            dom->trace(this->queue[domTarget], moved_rays);
                        }

                        boost::atomic<int> current_ray(0);
                        size_t workload = std::max((size_t) 1, (size_t) (moved_rays.size() / (GVT::Concurrency::asyncExec::instance()->numThreads * 2)));
                        {
                            boost::timer::auto_cpu_timer t("Scheduling rays %t\n");
                            for (int rc = 0; rc < GVT::Concurrency::asyncExec::instance()->numThreads; ++rc) {
                                GVT::Concurrency::asyncExec::instance()->run_task(processRayVector(this, moved_rays, current_ray, moved_rays.size(), workload, dom));
                            }
                            GVT::Concurrency::asyncExec::instance()->sync();
                        }
                        //                        while (!moved_rays.empty()) {
                        //                            GVT::Data::ray& mr = moved_rays.back();
                        //                            if(!mr.domains.empty()) {
                        //                                int target = mr.domains.back();
                        //                                this->queue[target].push_back(mr);
                        //                            }
                        //                            if(mr.type != GVT::Data::ray::PRIMARY) {
                        //                                this->addRay(mr);
                        //                            }
                        //                            moved_rays.pop_back();
                        //                        }

                        this->queue.erase(domTarget); // TODO: for secondary rays, rays may have been added to this domain queue
                    }


                }
                SUDO_DEBUG( else if (DEBUG_RANK) cerr << "skipped queues.  empty:" << this->queue.empty() << " size:" << this->queue.size() << endl);
                return true;
            }

            virtual bool SendRays() {
                int* outbound = new int[2 * this->world_size];
                int* inbound = new int[2 * this->world_size];
                MPI_Request* reqs = new MPI_Request[2 * this->world_size];
                MPI_Status* stat = new MPI_Status[2 * this->world_size];
                std::map<int, int> doms_to_send;
                int tag = 0;

                // init bufs
                for (int i = 0; i < 2 * this->world_size; ++i) {
                    inbound[i] = outbound[i] = 0;
                    reqs[i] = MPI_REQUEST_NULL;
                }

                // send to master current domain and ray counts for other domains
                DEBUG(if (DEBUG_RANK) cerr << this->rank << ": sending current data map" << endl);
                int to_send = 2 * this->queue.size() + 1;
                int *map_size_buf = (this->rank == 0) ? new int[this->world_size] : NULL;
                int *map_send_buf = (this->rank == 0) ? NULL : new int[to_send];
                int **map_recv_bufs = (this->rank == 0) ? new int*[this->world_size] : NULL;
                int *data_send_buf = NULL;
                DEBUG(if (DEBUG_RANK) cerr << this->rank << ": to send is " << to_send << endl);
                MPI_Gather(&to_send, 1, MPI_INT, map_size_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
                if (this->rank == 0) {
                    DEBUG(if (DEBUG_RANK) {
                        cerr << this->rank << ": map_size_buf:";
                        for (int i = 0; i < this->world_size; ++i) cerr << map_size_buf[i] << " ";
                                cerr << endl;
                        });
                    // add self
                    map_recv_bufs[0] = NULL;
                    if (to_send > 1) {
                        int ptr = 0;
                        map_recv_bufs[0] = new int[to_send];
                        map_recv_bufs[0][ptr++] = domTarget;
                        for (map<int, GVT::Data::RayVector>::iterator q = this->queue.begin(); q != this->queue.end(); ++q) {
                            map_recv_bufs[0][ptr++] = q->first;
                            map_recv_bufs[0][ptr++] = q->second.size();
                        }
                        DEBUG(if (DEBUG_RANK) {
                            cerr << this->rank << ": current rays (domTarget " << domTarget << ")" << endl;
                            for (int i = 1; i < to_send; i += 2)
                                    cerr << "    dom " << map_recv_bufs[0][i] << " (" << map_recv_bufs[0][i + 1] << " rays)" << endl;
                            });
                    }
                    for (int s = 1; s < this->world_size; ++s) // don't recv from self
                    {
                        map_recv_bufs[s] = NULL;
                        if (map_size_buf[s] > 1) {
                            map_recv_bufs[s] = new int[map_size_buf[s]];
                            MPI_Irecv(map_recv_bufs[s], map_size_buf[s], MPI_INT, s, tag, MPI_COMM_WORLD, &reqs[2 * s]);
                        }
                    }
                } else {
                    if (to_send > 1) {
                        int ptr = 0;
                        map_send_buf[ptr++] = domTarget;
                        for (map<int, GVT::Data::RayVector>::iterator q = this->queue.begin(); q != this->queue.end(); ++q) {
                            map_send_buf[ptr++] = q->first;
                            map_send_buf[ptr++] = q->second.size();
                        }
                        DEBUG(if (DEBUG_RANK) {
                            cerr << this->rank << ": current rays (domTarget " << domTarget << ")" << endl;
                            for (int i = 1; i < to_send; i += 2)
                                    cerr << "    dom " << map_send_buf[i] << " (" << map_send_buf[i + 1] << " rays)" << endl;
                            });
                        MPI_Isend(map_send_buf, to_send, MPI_INT, 0, tag, MPI_COMM_WORLD, &reqs[2 * this->rank + 1]);
                    }
                }
                DEBUG(if (DEBUG_RANK) cerr << this->rank << ": current data map sent" << endl);
                MPI_Waitall(2 * this->world_size, reqs, stat);
                DEBUG(if (DEBUG_RANK) cerr << this->rank << ": sync" << endl);

                // make new data map
                int *newMap = NULL;
                if (this->rank == 0) {
                    DEBUG(if (DEBUG_RANK) cerr << this->rank << ": making new data map" << endl);
                    data_send_buf = new int[this->world_size];
                    newMap = new int[this->world_size];
                    for (int i = 0; i < this->world_size; ++i)
                        data_send_buf[i] = -1;

                    SCHEDULER(newMap, this->world_size, map_size_buf, map_recv_bufs, data_send_buf)();
                }

                // send new map to procs
                DEBUG(if (DEBUG_RANK) cerr << this->rank << ": sending new data map" << endl);
                for (int i = 0; i < 2 * this->world_size; ++i) reqs[i] = MPI_REQUEST_NULL;
                tag = tag + 1;
                if (this->rank == 0) {
                    for (int s = 1; s < this->world_size; ++s) {
                        MPI_Isend(newMap, this->world_size, MPI_INT, s, tag, MPI_COMM_WORLD, &reqs[2 * s]);
                    }
                } else {
                    newMap = new int[this->world_size];
                    MPI_Irecv(newMap, this->world_size, MPI_INT, 0, tag, MPI_COMM_WORLD, &reqs[2 * this->rank + 1]);
                }
                DEBUG(if (DEBUG_RANK) cerr << this->rank << ": new data map sent" << endl);
                MPI_Waitall(2 * this->world_size, reqs, stat);
                DEBUG(if (DEBUG_RANK) cerr << this->rank << ": sync" << endl);

                // update data map and domains-to-send
                DEBUG(if (DEBUG_RANK) cerr << this->rank << ": updating local data map" << endl);
                for (int i = 0; i < this->world_size; ++i) {
                    if (i == this->rank) continue;

                    DEBUG(if (DEBUG_RANK) cerr << "    considering " << i << " -> " << newMap[i] << endl);
                    if (newMap[i] != newMap[this->rank]
                            && this->queue.find(newMap[i]) != this->queue.end()) {
                        // if there's no proc in the domain to send map, add it
                        // else, flip a coin to see if the old one is replaced
                        if (doms_to_send.find(newMap[i]) == doms_to_send.end())
                            doms_to_send[newMap[i]] = i;
                        else if (rand() & 0x1) {
                            DEBUG(if (DEBUG_RANK) cerr << "    doms_to_send was " << doms_to_send[newMap[i]] << endl);
                            doms_to_send[newMap[i]] = i;
                        }
                        DEBUG(if (DEBUG_RANK) cerr << "    doms_to_send now " << doms_to_send[newMap[i]] << endl);
                    }
                }

                // send dataset to procs
                DEBUG(if (DEBUG_RANK) cerr << this->rank << ": sending dataset send buf" << endl);
                for (int i = 0; i < 2 * this->world_size; ++i) reqs[i] = MPI_REQUEST_NULL;
                tag = tag + 1;
                if (this->rank == 0) {
                    for (int s = 1; s < this->world_size; ++s) {
                        MPI_Isend(data_send_buf, this->world_size, MPI_INT, s, tag, MPI_COMM_WORLD, &reqs[2 * s]);
                    }
                } else {
                    data_send_buf = new int[this->world_size];
                    MPI_Irecv(data_send_buf, this->world_size, MPI_INT, 0, tag, MPI_COMM_WORLD, &reqs[2 * this->rank + 1]);
                }
                DEBUG(if (DEBUG_RANK) cerr << this->rank << ": new dataset send buf sent" << endl);
                MPI_Waitall(2 * this->world_size, reqs, stat);
                DEBUG(if (DEBUG_RANK) cerr << this->rank << ": sync" << endl);

                for (int i = 0; i < 2 * this->world_size; ++i) reqs[i] = MPI_REQUEST_NULL;
                // if the value for this proc >= 0, then receive from value proc
                // if this proc in a value, send to that index (may be more than one)
#ifdef SEND_DOMS
#undef SEND_DOMS  // pnav: don't send any domains right now
#endif
#if SEND_DOMS
                int dom_char_send_len = 0,
                        dom_char_recv_len = 0;
                char ** dom_char_send = NULL, // MPI currently doesn't support concurrent buf readom, so must dup
                        * dom_char_recv = NULL;
                vector<int> sendto;

                // figure out if we're sending to anyone
                for (int i = 0; i < size; ++i)
                    if (data_send_buf[i] == rank)
                        sendto.push_back(i);

                dom_char_send = new char*[sendto.size()];
                tag = tag + 1;
                // first send and receive the dataset size, if needed
                if (data_send_buf[rank] >= 0) {
                    MPI_Irecv(&dom_char_recv_len, 1, MPI_INT, data_send_buf[rank], tag, MPI_COMM_WORLD, &reqs[2 * rank + 1]);
                }

                if (!sendto.empty()) {
                    DEBUG(if (DEBUG_RANK) cerr << rank << ": serializing dataset " << newMap[rank] << endl);
                    if (newMap[rank] != domTarget) {
                        DEBUG(if (DEBUG_RANK) cerr << "    must swap datasets.  Had " << domTarget << " but need to send " << newMap[rank] << endl);
                        // bugger! gotta load the data then send to the rest
                        if (dom != NULL) dom->UnRegister(NULL);
                        dom = rta.dataset.GetDomain(newMap[rank]);
                        dom_mailbox = dom; // so we don't call GetDomain() again
                        dom->Register(NULL);
                        // track domain loadom
                        ++domain_counter;
                    }

                    vtkDataSetWriter *writer = vtkDataSetWriter::New();
                    writer->SetInput(dom);
                    writer->SetWriteToOutputString(1);
                    writer->SetFileTypeToBinary();
                    writer->Write();
                    writer->Update();
                    dom_char_send_len = writer->GetOutputStringLength();
                    dom_char_send[0] = writer->RegisterAndGetOutputString();
                    writer->Delete();
                    DEBUG(if (DEBUG_RANK) {
                        cerr << "    done.  " << dom->GetNumberOfCells() << " cells, char length " << dom_char_send_len << " starting at " << (void*) dom_char_send[0] << endl;
                                cerr << "        first 10 chars: ";
                        for (int i = 0; i < 10; ++i)
                                cerr << (int) dom_char_send[0][i] << " ";
                                cerr << endl;
                        });

                    for (int i = 0; i < sendto.size(); ++i)
                        MPI_Isend(&dom_char_send_len, 1, MPI_INT, sendto[i], tag, MPI_COMM_WORLD, &reqs[2 * rank]);
                }
                DEBUG(if (DEBUG_RANK) cerr << rank << ": dataset buf lengths sent" << endl);
                MPI_Waitall(2 * size, reqs, stat);
                DEBUG(if (DEBUG_RANK) cerr << rank << ": sync" << endl);

                tag = tag + 1;
                for (int i = 0; i < 2 * size; ++i) reqs[i] = MPI_REQUEST_NULL;
                // now send and receive the data itself
                if (data_send_buf[rank] >= 0) {
                    DEBUG(if (DEBUG_RANK) cerr << rank << ": receiving dataset " << newMap[rank] << " of length " << dom_char_recv_len << endl);
                    dom_char_recv = new char[dom_char_recv_len];
                    MPI_Irecv(dom_char_recv, dom_char_recv_len, MPI_CHAR, data_send_buf[rank], tag, MPI_COMM_WORLD, &reqs[2 * rank + 1]);
                }

                if (!sendto.empty()) {
                    for (int i = 1; i < sendto.size(); ++i) {
                        dom_char_send[i] = new char[dom_char_send_len];
                        memcpy(dom_char_send[i], dom_char_send[0], dom_char_send_len);
                        DEBUG(if (DEBUG_RANK) {
                            cerr << "        copy " << i << " first 10 chars: ";
                            for (int ii = 0; ii < 10; ++ii)
                                    cerr << (int) dom_char_send[i][ii] << " ";
                                    cerr << endl;
                            });
                    }
                    for (int i = 0; i < sendto.size(); ++i) {
                        DEBUG(if (DEBUG_RANK) cerr << rank << ": sending dataset size to " << i << endl);
                        MPI_Isend(dom_char_send[i], dom_char_send_len, MPI_CHAR, sendto[i], tag, MPI_COMM_WORLD, &reqs[2 * rank]);
                    }
                }
                DEBUG(if (DEBUG_RANK) cerr << rank << ": dataset send and recv bufs sent" << endl);
                MPI_Waitall(2 * size, reqs, stat);
                DEBUG(if (DEBUG_RANK) cerr << rank << ": sync" << endl);

                if (data_send_buf[rank] >= 0) {
                    DEBUG(if (DEBUG_RANK) cerr << rank << ": unserializing new dataset" << endl);
                    DEBUG(if (DEBUG_RANK) {
                        cerr << "        first 10 chars: ";
                        for (int i = 0; i < 10; ++i)
                                cerr << (int) dom_char_recv[i] << " ";
                                cerr << endl;
                        });
#if 0
                    vtkDataSetReader * reader = vtkDataSetReader::New();
                    vtkCharArray * dom_ca = vtkCharArray::New();

                    dom_ca->SetArray(dom_char_recv, dom_char_recv_len, 1);
                    reader->SetReadFromInputString(1);
                    reader->SetInputArray(dom_ca);
                    dom_mailbox = reader->GetOutput();
                    //dom_mailbox->Update();
                    dom_mailbox->Register(NULL);
                    reader->Delete(); // delete reader first
                    dom_ca->Delete();
#else
                    vtkRectilinearGridReader *reader = vtkRectilinearGridReader::New();
                    vtkCharArray * dom_ca = vtkCharArray::New();

                    dom_mailbox = (vtkDataSet*) reader->GetOutput();
                    dom_ca->SetArray(dom_char_recv, dom_char_recv_len, 1);
                    reader->SetReadFromInputString(1);
                    reader->SetInputArray(dom_ca);
                    dom_mailbox->Update();
                    dom_mailbox->Register(NULL);
                    reader->Delete();
                    dom_ca->Delete();
#endif
                    DEBUG(if (DEBUG_RANK) cerr << rank << ": unserialized dataset to mailbox at " << dom_mailbox << " with " << dom_mailbox->GetNumberOfCells() << " cells" << endl);
                }
#endif /* SEND_DOMS */

                // assign new domain target
                DEBUG(if (DEBUG_RANK) cerr << this->rank << ": domain was " << domTarget << " and is now " << newMap[this->rank] << endl);
                domTarget = newMap[this->rank];

                // we (hopefully) divied up target procs across source procs
                // so blindly obey the dos
                // XXX TODO: some smarter ray division?
                for (std::map<int, int>::iterator dos = doms_to_send.begin(); dos != doms_to_send.end(); ++dos) {
                    int ray_count = 0,
                            ray_seg = this->queue[dos->first].size();

                    int i = dos->second * 2;
                    outbound[i] += ray_seg;
                    int buf_size = 0;
                    DEBUG(if (DEBUG_RANK) cerr << this->rank << ": ray sizes in domain " << dos->first << endl);
                    for (int r = ray_count; r < (ray_count + ray_seg); ++r) {
                        buf_size += this->queue[dos->first][r].packedSize(); // rays can have diff packed sizes
                        DEBUG(if (DEBUG_RANK) cerr << "    " << this->queue[dos->first][r]->packedSize() << " (" << buf_size << ")" << endl);
                    }
                    outbound[i + 1] += buf_size;
                    ray_count += ray_seg;
                }

                // let the targets know what's coming
                // and find out what's coming here
                DEBUG(if (DEBUG_RANK) cerr << this->rank << ": sending target info" << endl);
                MPI_Alltoall(outbound, 2, MPI_INT, inbound, 2, MPI_INT, MPI_COMM_WORLD);
                DEBUG(if (DEBUG_RANK) {
                    cerr << this->rank << ": sent target info" << endl;
                            cerr << this->rank << ": inbound ";
                    for (int i = 0; i < this->world_size; ++i)
                            cerr << "(" << inbound[2 * i] << "," << inbound[2 * i + 1] << ") ";
                            cerr << endl << this->rank << ": outbound ";
                        for (int i = 0; i < this->world_size; ++i)
                                cerr << "(" << outbound[2 * i] << "," << outbound[2 * i + 1] << ") ";
                                cerr << endl;
                        });

                // set up send and recv buffers
                unsigned char** send_buf = new unsigned char*[this->world_size];
                unsigned char** recv_buf = new unsigned char*[this->world_size];
                int *ptr_buf = new int[this->world_size];
                for (int i = 0, j = 0; i < this->world_size; ++i, j += 2) {
                    if (outbound[j] > 0)
                        send_buf[i] = new unsigned char[outbound[j + 1]];
                    else send_buf[i] = 0;
                    if (inbound[j] > 0)
                        recv_buf[i] = new unsigned char[inbound[j + 1]];
                    else recv_buf[i] = 0;
                    ptr_buf[i] = 0;
                }

                // now send and receive rays (and associated color buffers)
                for (int i = 0; i < 2 * this->world_size; ++i) reqs[i] = MPI_REQUEST_NULL;
                tag = tag + 1;
                for (int i = 0, j = 0; i < this->world_size; ++i, j += 2) {
                    if (inbound[j] > 0) {
                        DEBUG(if (DEBUG_RANK)
                                cerr << this->rank << ": recv " << inbound[j] << " rays ("
                                << inbound[j + 1] << " bytes) from " << i << endl
                                );
                        MPI_Irecv(recv_buf[i], inbound[j + 1], MPI_UNSIGNED_CHAR, i, tag, MPI_COMM_WORLD, &reqs[j]);
                    }
                }

                for (std::map<int, int>::iterator dos = doms_to_send.begin(); dos != doms_to_send.end(); ++dos) {
                    int ray_count = 0,
                            ray_seg = this->queue[dos->first].size();

                    int i = dos->second * 2;
                    DEBUG(if (DEBUG_RANK)
                            cerr << this->rank << ": send " << outbound[i] << " rays ("
                            << outbound[i + 1] << " bytes) to " << dos->second << endl;
                            );
                    for (int r = ray_count; r < (ray_count + ray_seg); ++r) {
                        GVT::Data::ray& ray = this->queue[dos->first][r];
                        DEBUG(if (DEBUG_RANK) cerr << "    @" << ptr_buf[dos->second] << ": " << ray << endl);
                        ptr_buf[dos->second] += ray.pack(send_buf[dos->second] + ptr_buf[dos->second]);
                    }
                    ray_count += ray_seg;

                    DEBUG(if (DEBUG_RANK) cerr << this->rank << ": q(" << this->queue.size() << ") erasing " << dos->first);
                    this->queue.erase(dos->first);
                    DEBUG(if (DEBUG_RANK) cerr << " q(" << this->queue.size() << ")" << endl);
                }

                for (int i = 0, j = 0; i < this->world_size; ++i, j += 2) {
                    if (outbound[j] > 0) {
                        DEBUG(if (DEBUG_RANK) cerr << this->rank << ": sending " << ptr_buf[i] << " bytes to " << i << endl);
                        MPI_Isend(send_buf[i], outbound[j + 1], MPI_UNSIGNED_CHAR, i, tag, MPI_COMM_WORLD, &reqs[j + 1]);
                    }
                }
                MPI_Waitall(2 * this->world_size, reqs, stat); // XXX TODO refactor to use Waitany?

                for (int i = 0, j = 0; i < this->world_size; ++i, j += 2) {
                    if (inbound[j] > 0) {
                        DEBUG(if (DEBUG_RANK) {
                            cerr << this->rank << ": adding " << inbound[j] <<
                                    " rays (" << inbound[j + 1] << " B) from " << i << endl;
                                    cerr << "    recv buf: " << (void *) recv_buf[i] << endl;
                        });
                        int ptr = 0;
                        for (int c = 0; c < inbound[j]; ++c) {
                            DEBUG(if (DEBUG_RANK) cerr << "    receive ray " << c << endl);
                            GVT::Data::ray r(recv_buf[i] + ptr);
                            this->queue[r.domains.back()].push_back(r);
                            ptr += r.packedSize();
                            DEBUG(if (DEBUG_RANK) cerr << "    " << r << endl);
                        }
                    }
                }
                DEBUG(if (DEBUG_RANK) cerr << this->rank << ": sent and received rays" << endl);

                // clean up
                doms_to_send.clear();
                for (int i = 0; i < this->world_size; ++i) {
                    delete[] send_buf[i];
                    delete[] recv_buf[i];
                }
                if (map_recv_bufs != NULL)
                    for (int i = 0; i < this->world_size; ++i)
                        delete[] map_recv_bufs[i];
#ifdef SEND_DOMS
                for (int i = 0; i < sendto.size(); ++i)
                    delete[] dom_char_send[i];
#endif
                delete[] send_buf;
                delete[] recv_buf;
                delete[] ptr_buf;
                delete[] map_size_buf;
                delete[] map_send_buf;
                delete[] data_send_buf;
                delete[] map_recv_bufs;
                delete[] newMap;
#ifdef SEND_DOMS
                delete[] dom_char_send;
                delete[] dom_char_recv;
#endif
                delete[] reqs;
                delete[] stat;
                delete[] inbound;
                delete[] outbound;
                DEBUG(if (DEBUG_RANK) cerr << this->rank << ": cleaned up" << endl);
                return true;
            }
        };
    };
};

#endif	/* GVT_HYBRIDTRACER_H */

