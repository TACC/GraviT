#if 0
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
 * File:   HybridTracer.h
 * Author: jbarbosa
 *
 * Created on January 21, 2014, 10:46 AM
 */

#ifndef GVT_RENDER_ALGORITHM_HYBRID_TRACER_H
#define GVT_RENDER_ALGORITHM_HYBRID_TRACER_H

#include <gvt/core/Debug.h>
#include <gvt/core/mpi/Wrapper.h>
#include <gvt/render/Schedulers.h>
#include <gvt/render/algorithm/TracerBase.h>

#include <boost/foreach.hpp>

#include <mpi.h>

namespace gvt {
namespace render {
namespace algorithm {

/// work scheduler that adapts between Image and Domain scheduling based on render state
/**
  The Hybrid scheduler strives to combine the advantages of Image and Domain scheduling
  to achieve performant work balance across the system for the entire render process. The
  initial schedule divides rays across processes, similar to the Image scheduler, then
  operates according to the heuristics defined in the gvt::render::schedule::hybrid classes.
  These typically seek to maintain loaded domains in-core so long as there are rays that need
  the domain's data.

  This scheduler can become unbalanced depending on the characteristics of the schedule heuristics.

    \sa DomainTracer, ImageTracer
   */
template <class SCHEDULER> class Tracer<gvt::render::schedule::HybridScheduler<SCHEDULER> > : public AbstractTrace {
public:
  size_t rays_start, rays_end;

  gvt::render::data::domain::AbstractDomain *dom;
  int domTarget;
  long domain_counter;

  Tracer(gvt::render::actor::RayVector &rays, gvt::render::data::scene::Image &image) : AbstractTrace(rays, image) {
    int ray_portion = this->rays.size() / mpi.world_size;
    this->rays_start = mpi.rank * ray_portion;
    this->rays_end = (mpi.rank + 1) == mpi.world_size
                         ? this->rays.size()
                         : (mpi.rank + 1) * ray_portion; // tack on any odd rays to last proc
  }

  ~Tracer() {}

  virtual void operator()() {


                                                      << " rays" << std::endl);

    long ray_counter = 0;



    this->FilterRaysLocally();



    // process domains until all rays are terminated
    bool all_done = false;
    gvt::core::Map<int, int> doms_to_send;
    int domTargetCount = 0, lastDomain = -1;
    gvt::render::data::domain::AbstractDomain *dom_mailbox = NULL;
    gvt::render::actor::RayVector moved_rays;
    moved_rays.reserve(1000);

    // begin by process domain with most rays queued
    for (gvt::core::Map<int, gvt::render::actor::RayVector>::iterator q = this->queue.begin(); q != this->queue.end(); ++q) {
      if (q->second.size() > domTargetCount) {
        domTargetCount = q->second.size();
        domTarget = q->first;
      }
    }


                                                      << " rays)" << std::endl);

    while (!all_done) {

      schedule(this->queue, domTarget, lastDomain, dom_mailbox, moved_rays, ray_counter);
      // done with current domain, send off rays to their proper processors.

      SendRays();

      // are we done?

      // root proc takes empty flag from all procs
      int not_done = (int)(!this->queue.empty());
      int *empties = (mpi.rank == 0) ? new int[mpi.world_size] : NULL;
      MPI_Gather(&not_done, 1, MPI_INT, empties, 1, MPI_INT, 0, MPI_COMM_WORLD);

      if (mpi.rank == 0) {
        not_done = 0;
        for (int i = 0; i < mpi.world_size; ++i) not_done += empties[i];
        for (int i = 0; i < mpi.world_size; ++i) empties[i] = not_done;
      }

      MPI_Scatter(empties, 1, MPI_INT, &not_done, 1, MPI_INT, 0, MPI_COMM_WORLD);


                                                        << " (my q:" << this->queue.size() << ")" << std::endl);

          DBG_LOW, if (DEBUG_RANK) for (gvt::core::Map<int, gvt::render::actor::RayVector>::iterator q = this->queue.begin();
                                        q != this->queue.end(); ++q) std::cerr
                       << "    q(" << q->first << "):" << q->second.size() << std::endl);

      all_done = (not_done == 0);

      delete[] empties;
    }

    this->gatherFramebuffers(this->rays.size());
  }

  virtual bool schedule(gvt::core::Map<int, gvt::render::actor::RayVector> &queue, int &domTarget, int &lastDomain,
                        gvt::render::data::domain::AbstractDomain *dom_mailbox,
                        gvt::render::actor::RayVector &moved_rays, long &ray_counter) {
    dom = NULL;
    if (!this->queue.empty()) {
      // pnav: use this to ignore domain x:        int domi=0;if (0)
      if (domTarget >= 0) {



        if (domTarget != lastDomain)
          if (dom != NULL) dom->free();

        // register the dataset so it doesn't get deleted
        // if we build a lightmap
        if (dom_mailbox != NULL) {
          dom = dom_mailbox;
          dom_mailbox = NULL;
          // was registered when it was loaded

          // don't increment counter here, since it wasn't loaded from disk
          if (domTarget != lastDomain) {
            lastDomain = domTarget;
          }
        } else {
          dom = gvt::render::Attributes::rta->dataset->getDomain(domTarget);

                                                            << std::endl);
          // track domain loads
          if (domTarget != lastDomain) {
            ++domain_counter;
            lastDomain = domTarget;
            dom->load();
          }
        }

        // GVT::Backend::ProcessQueue<DomainType>(new GVT::Backend::adapt_param<DomainType>(this->queue, moved_rays,
        // domTarget, dom, this->colorBuf, ray_counter, domain_counter))();
        {
          moved_rays.reserve(this->queue[domTarget].size() * 10);
          boost::timer::auto_cpu_timer t("Tracing domain rays %t\n");
          dom->trace(this->queue[domTarget], moved_rays);
          this->queue[domTarget].clear();
        }

        boost::atomic<int> current_ray(0);
        size_t workload = std::max(
            (size_t)1, (size_t)(moved_rays.size() / (gvt::core::schedule::asyncExec::instance()->numThreads * 2)));
        {
          boost::timer::auto_cpu_timer t("Scheduling rays %t\n");
          for (int rc = 0; rc < gvt::core::schedule::asyncExec::instance()->numThreads; ++rc) {
            gvt::core::schedule::asyncExec::instance()->run_task(
                processRayVector(this, moved_rays, current_ray, moved_rays.size(), workload, dom));
          }
          gvt::core::schedule::asyncExec::instance()->sync();
        }
        //                        while (!moved_rays.empty()) {
        //                            gvt::render::actor::Ray& mr = moved_rays.back();
        //                            if(!mr.domains.empty()) {
        //                                int target = mr.domains.back();
        //                                this->queue[target].push_back(mr);
        //                            }
        //                            if(mr.type != gvt::render::actor::Ray::PRIMARY) {
        //                                this->addRay(mr);
        //                            }
        //                            moved_rays.pop_back();
        //                        }

        this->queue.erase(domTarget); // TODO: for secondary rays, rays may have been added to this domain queue
      }
    }
    // SUDO_DEBUG( else if (DEBUG_RANK) std::cerr << "skipped queues.  empty:" << this->queue.empty() << " size:" <<
    // this->queue.size() << std::endl);
    return true;
  }

  virtual bool SendRays() {
    int *outbound = new int[2 * mpi.world_size];
    int *inbound = new int[2 * mpi.world_size];
    MPI_Request *reqs = new MPI_Request[2 * mpi.world_size];
    MPI_Status *stat = new MPI_Status[2 * mpi.world_size];
    gvt::core::Map<int, int> doms_to_send;
    int tag = 0;

    // init bufs
    for (int i = 0; i < 2 * mpi.world_size; ++i) {
      inbound[i] = outbound[i] = 0;
      reqs[i] = MPI_REQUEST_NULL;
    }

    // send to master current domain and ray counts for other domains

    int to_send = 2 * this->queue.size() + 1;
    int *map_size_buf = (mpi.rank == 0) ? new int[mpi.world_size] : NULL;
    int *map_send_buf = (mpi.rank == 0) ? NULL : new int[to_send];
    int **map_recv_bufs = (mpi.rank == 0) ? new int *[mpi.world_size] : NULL;
    int *data_send_buf = NULL;

    MPI_Gather(&to_send, 1, MPI_INT, map_size_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (mpi.rank == 0) {

        std::cerr << mpi.rank << ": map_size_buf:";
        for (int i = 0; i < mpi.world_size; ++i) std::cerr << map_size_buf[i] << " ";
        std::cerr << std::endl;
      });
      // add self
      map_recv_bufs[0] = NULL;
      if (to_send > 1) {
        int ptr = 0;
        map_recv_bufs[0] = new int[to_send];
        map_recv_bufs[0][ptr++] = domTarget;
        for (gvt::core::Map<int, gvt::render::actor::RayVector>::iterator q = this->queue.begin(); q != this->queue.end();
             ++q) {
          map_recv_bufs[0][ptr++] = q->first;
          map_recv_bufs[0][ptr++] = q->second.size();
        }

          std::cerr << mpi.rank << ": current rays (domTarget " << domTarget << ")" << std::endl;
          for (int i = 1; i < to_send; i += 2)
            std::cerr << "    dom " << map_recv_bufs[0][i] << " (" << map_recv_bufs[0][i + 1] << " rays)" << std::endl;
        });
      }
      for (int s = 1; s < mpi.world_size; ++s) // don't recv from self
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
        for (gvt::core::Map<int, gvt::render::actor::RayVector>::iterator q = this->queue.begin(); q != this->queue.end();
             ++q) {
          map_send_buf[ptr++] = q->first;
          map_send_buf[ptr++] = q->second.size();
        }

          std::cerr << mpi.rank << ": current rays (domTarget " << domTarget << ")" << std::endl;
          for (int i = 1; i < to_send; i += 2)
            std::cerr << "    dom " << map_send_buf[i] << " (" << map_send_buf[i + 1] << " rays)" << std::endl;
        });
        MPI_Isend(map_send_buf, to_send, MPI_INT, 0, tag, MPI_COMM_WORLD, &reqs[2 * mpi.rank + 1]);
      }
    }

    MPI_Waitall(2 * mpi.world_size, reqs, stat);


    // make new data map
    int *newMap = NULL;
    if (mpi.rank == 0) {

      data_send_buf = new int[mpi.world_size];
      newMap = new int[mpi.world_size];
      for (int i = 0; i < mpi.world_size; ++i) data_send_buf[i] = -1;

      SCHEDULER(newMap, mpi.world_size, map_size_buf, map_recv_bufs, data_send_buf)();
    }

    // send new map to procs

    for (int i = 0; i < 2 * mpi.world_size; ++i) reqs[i] = MPI_REQUEST_NULL;
    tag = tag + 1;
    if (mpi.rank == 0) {
      for (int s = 1; s < mpi.world_size; ++s) {
        MPI_Isend(newMap, mpi.world_size, MPI_INT, s, tag, MPI_COMM_WORLD, &reqs[2 * s]);
      }
    } else {
      newMap = new int[mpi.world_size];
      MPI_Irecv(newMap, mpi.world_size, MPI_INT, 0, tag, MPI_COMM_WORLD, &reqs[2 * mpi.rank + 1]);
    }

    MPI_Waitall(2 * mpi.world_size, reqs, stat);


    // update data map and domains-to-send

    for (int i = 0; i < mpi.world_size; ++i) {
      if (i == mpi.rank) continue;


      if (newMap[i] != newMap[mpi.rank] && this->queue.find(newMap[i]) != this->queue.end()) {
        // if there's no proc in the domain to send map, add it
        // else, flip a coin to see if the old one is replaced
        if (doms_to_send.find(newMap[i]) == doms_to_send.end())
          doms_to_send[newMap[i]] = i;
        else if (rand() & 0x1) {

                                                            << std::endl);
          doms_to_send[newMap[i]] = i;
        }

                                                          << std::endl);
      }
    }

    // send dataset to procs

    for (int i = 0; i < 2 * mpi.world_size; ++i) reqs[i] = MPI_REQUEST_NULL;
    tag = tag + 1;
    if (mpi.rank == 0) {
      for (int s = 1; s < mpi.world_size; ++s) {
        MPI_Isend(data_send_buf, mpi.world_size, MPI_INT, s, tag, MPI_COMM_WORLD, &reqs[2 * s]);
      }
    } else {
      data_send_buf = new int[mpi.world_size];
      MPI_Irecv(data_send_buf, mpi.world_size, MPI_INT, 0, tag, MPI_COMM_WORLD, &reqs[2 * mpi.rank + 1]);
    }

    MPI_Waitall(2 * mpi.world_size, reqs, stat);


    for (int i = 0; i < 2 * mpi.world_size; ++i) reqs[i] = MPI_REQUEST_NULL;
// if the value for this proc >= 0, then receive from value proc
// if this proc in a value, send to that index (may be more than one)
#ifdef SEND_DOMS
#undef SEND_DOMS // pnav: don't send any domains right now
#endif
#if SEND_DOMS
    int dom_char_send_len = 0, dom_char_recv_len = 0;
    char **dom_char_send = NULL, // MPI currently doesn't support concurrent buf readom, so must dup
        *dom_char_recv = NULL;
    vector<int> sendto;

    // figure out if we're sending to anyone
    for (int i = 0; i < size; ++i)
      if (data_send_buf[i] == rank) sendto.push_back(i);

    dom_char_send = new char *[sendto.size()];
    tag = tag + 1;
    // first send and receive the dataset size, if needed
    if (data_send_buf[rank] >= 0) {
      MPI_Irecv(&dom_char_recv_len, 1, MPI_INT, data_send_buf[rank], tag, MPI_COMM_WORLD, &reqs[2 * rank + 1]);
    }

    if (!sendto.empty()) {

                                                        << std::endl);
      if (newMap[rank] != domTarget) {

                                                          << " but need to send " << newMap[rank] << std::endl);
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

        std::cerr << "    done.  " << dom->GetNumberOfCells() << " cells, char length " << dom_char_send_len
                  << " starting at " << (void *)dom_char_send[0] << std::endl;
        std::cerr << "        first 10 chars: ";
        for (int i = 0; i < 10; ++i) std::cerr << (int)dom_char_send[0][i] << " ";
        std::cerr << std::endl;
      });

      for (int i = 0; i < sendto.size(); ++i)
        MPI_Isend(&dom_char_send_len, 1, MPI_INT, sendto[i], tag, MPI_COMM_WORLD, &reqs[2 * rank]);
    }

    MPI_Waitall(2 * size, reqs, stat);


    tag = tag + 1;
    for (int i = 0; i < 2 * size; ++i) reqs[i] = MPI_REQUEST_NULL;
    // now send and receive the data itself
    if (data_send_buf[rank] >= 0) {

                                                        << " of length " << dom_char_recv_len << std::endl);
      dom_char_recv = new char[dom_char_recv_len];
      MPI_Irecv(dom_char_recv, dom_char_recv_len, MPI_CHAR, data_send_buf[rank], tag, MPI_COMM_WORLD,
                &reqs[2 * rank + 1]);
    }

    if (!sendto.empty()) {
      for (int i = 1; i < sendto.size(); ++i) {
        dom_char_send[i] = new char[dom_char_send_len];
        memcpy(dom_char_send[i], dom_char_send[0], dom_char_send_len);

          std::cerr << "        copy " << i << " first 10 chars: ";
          for (int ii = 0; ii < 10; ++ii) std::cerr << (int)dom_char_send[i][ii] << " ";
          std::cerr << std::endl;
        });
      }
      for (int i = 0; i < sendto.size(); ++i) {

        MPI_Isend(dom_char_send[i], dom_char_send_len, MPI_CHAR, sendto[i], tag, MPI_COMM_WORLD, &reqs[2 * rank]);
      }
    }

    MPI_Waitall(2 * size, reqs, stat);


    if (data_send_buf[rank] >= 0) {


        std::cerr << "        first 10 chars: ";
        for (int i = 0; i < 10; ++i) std::cerr << (int)dom_char_recv[i] << " ";
        std::cerr << std::endl;
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
      vtkCharArray *dom_ca = vtkCharArray::New();

      dom_mailbox = (vtkDataSet *)reader->GetOutput();
      dom_ca->SetArray(dom_char_recv, dom_char_recv_len, 1);
      reader->SetReadFromInputString(1);
      reader->SetInputArray(dom_ca);
      dom_mailbox->Update();
      dom_mailbox->Register(NULL);
      reader->Delete();
      dom_ca->Delete();
#endif

                                                        << dom_mailbox << " with " << dom_mailbox->GetNumberOfCells()
                                                        << " cells" << std::endl);
    }
#endif /* SEND_DOMS */

    // assign new domain target

                                                      << newMap[mpi.rank] << std::endl);
    domTarget = newMap[mpi.rank];

    // we (hopefully) divied up target procs across source procs
    // so blindly obey the dos
    // XXX TODO: some smarter ray division?
    for (gvt::core::Map<int, int>::iterator dos = doms_to_send.begin(); dos != doms_to_send.end(); ++dos) {
      int ray_count = 0, ray_seg = this->queue[dos->first].size();

      int i = dos->second * 2;
      outbound[i] += ray_seg;
      int buf_size = 0;

                                                        << std::endl);
      for (int r = ray_count; r < (ray_count + ray_seg); ++r) {
        buf_size += this->queue[dos->first][r].packedSize(); // rays can have diff packed sizes

                                                          << buf_size << ")" << std::endl);
      }
      outbound[i + 1] += buf_size;
      ray_count += ray_seg;
    }

    // let the targets know what's coming
    // and find out what's coming here

    MPI_Alltoall(outbound, 2, MPI_INT, inbound, 2, MPI_INT, MPI_COMM_WORLD);

      std::cerr << mpi.rank << ": sent target info" << std::endl;
      std::cerr << mpi.rank << ": inbound ";
      for (int i = 0; i < mpi.world_size; ++i) std::cerr << "(" << inbound[2 * i] << "," << inbound[2 * i + 1] << ") ";
      std::cerr << std::endl << mpi.rank << ": outbound ";
      for (int i = 0; i < mpi.world_size; ++i)
        std::cerr << "(" << outbound[2 * i] << "," << outbound[2 * i + 1] << ") ";
      std::cerr << std::endl;
    });

    // set up send and recv buffers
    unsigned char **send_buf = new unsigned char *[mpi.world_size];
    unsigned char **recv_buf = new unsigned char *[mpi.world_size];
    int *ptr_buf = new int[mpi.world_size];
    for (int i = 0, j = 0; i < mpi.world_size; ++i, j += 2) {
      if (outbound[j] > 0)
        send_buf[i] = new unsigned char[outbound[j + 1]];
      else
        send_buf[i] = 0;
      if (inbound[j] > 0)
        recv_buf[i] = new unsigned char[inbound[j + 1]];
      else
        recv_buf[i] = 0;
      ptr_buf[i] = 0;
    }

    // now send and receive rays (and associated color buffers)
    for (int i = 0; i < 2 * mpi.world_size; ++i) reqs[i] = MPI_REQUEST_NULL;
    tag = tag + 1;
    for (int i = 0, j = 0; i < mpi.world_size; ++i, j += 2) {
      if (inbound[j] > 0) {

                                                          << inbound[j + 1] << " bytes) from " << i << std::endl);
        MPI_Irecv(recv_buf[i], inbound[j + 1], MPI_UNSIGNED_CHAR, i, tag, MPI_COMM_WORLD, &reqs[j]);
      }
    }

    for (gvt::core::Map<int, int>::iterator dos = doms_to_send.begin(); dos != doms_to_send.end(); ++dos) {
      int ray_count = 0, ray_seg = this->queue[dos->first].size();

      int i = dos->second * 2;

                                                        << outbound[i + 1] << " bytes) to " << dos->second
                                                        << std::endl;);
      for (int r = ray_count; r < (ray_count + ray_seg); ++r) {
        gvt::render::actor::Ray &ray = this->queue[dos->first][r];

                                                          << std::endl);
        ptr_buf[dos->second] += ray.pack(send_buf[dos->second] + ptr_buf[dos->second]);
      }
      ray_count += ray_seg;


                                                        << dos->first);
      this->queue.erase(dos->first);

    }

    for (int i = 0, j = 0; i < mpi.world_size; ++i, j += 2) {
      if (outbound[j] > 0) {

                                                          << std::endl);
        MPI_Isend(send_buf[i], outbound[j + 1], MPI_UNSIGNED_CHAR, i, tag, MPI_COMM_WORLD, &reqs[j + 1]);
      }
    }
    MPI_Waitall(2 * mpi.world_size, reqs, stat); // XXX TODO refactor to use Waitany?

    for (int i = 0, j = 0; i < mpi.world_size; ++i, j += 2) {
      if (inbound[j] > 0) {

          std::cerr << mpi.rank << ": adding " << inbound[j] << " rays (" << inbound[j + 1] << " B) from " << i
                    << std::endl;
          std::cerr << "    recv buf: " << (void *)recv_buf[i] << std::endl;
        });
        int ptr = 0;
        for (int c = 0; c < inbound[j]; ++c) {

          gvt::render::actor::Ray r(recv_buf[i] + ptr);

          // TODO : Fix me

          // this->queue[r.domains.back()].push_back(r);
          ptr += r.packedSize();

        }
      }
    }


    // clean up
    doms_to_send.clear();
    for (int i = 0; i < mpi.world_size; ++i) {
      delete[] send_buf[i];
      delete[] recv_buf[i];
    }
    if (map_recv_bufs != NULL)
      for (int i = 0; i < mpi.world_size; ++i) delete[] map_recv_bufs[i];
#ifdef SEND_DOMS
    for (int i = 0; i < sendto.size(); ++i) delete[] dom_char_send[i];
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

    return true;
  }
};
}
}
}

#endif /* GVT_RENDER_ALGORITHM_HYBRID_TRACER_H */
#endif
