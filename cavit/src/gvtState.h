#ifndef GVTSTATE_H
#define GVTSTATE_H

#include <glm/vec3.hpp>
#include <glm/gtx/io.hpp>

// #include "tutorial/tutorial.h"

// a non-portable native binary archive
// boost::archive::binary_oarchive // saving
// boost::archive::binary_iarchive // loading
// include headers that implement a archive in simple text format
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <boost/config.hpp>
#include <boost/archive/detail/auto_link_archive.hpp>
#include <boost/archive/detail/register_archive.hpp>
#include <boost/archive/basic_text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
// #include <boost/mpi.hpp>

#include <mpi.h>

#include <sstream>
#include <vector>

typedef std::vector<char> MPIBuffer;

namespace cvt
{

  #define DEBUG(x) { std::cout << __PRETTY_FUNCTION__ << ":" << __LINE__ << ": \"" << #x << "\": " << x << std::endl; }

template<class T>
  void GetMPIState(T& state, MPI_Comm comm, int source, MPI_Datatype type, MPIBuffer& buffer, MPI_Status& status, int size=0)
  {
// Probe for an incoming message from process zero
  // printf("%s: probing for state from %d tag %d\n",__FUNCTION__, source, state.GetTag());
    if (size == 0)
    {
      MPI_Probe(source, state.GetTag(), comm, &status);

    // When probe returns, the status object has the size and other
    // attributes of the incoming message. Get the size of the message
      MPI_Get_count(&status, type, &size);
    }
  // char *buf = static_cast<char*>( malloc( size ) );
    if (buffer.size() < size)
      buffer.resize(size);
    MPI_Recv( &buffer[0], size, type, source,
      state.GetTag(), comm, &status );
    std::string s( &buffer[0], size );
    std::stringstream ss( s );
    boost::archive::text_iarchive ia(ss);
    ia >> state;
  // free(buf);
  // printf("%s: recieved state from %d\n",__FUNCTION__, source);
  }

template<class T>
  void SetMPIState(T& state, MPI_Comm comm, int dest, MPI_Datatype type, MPI_Request* request = NULL)
  {
    std::stringstream ss;
    {
      boost::archive::text_oarchive oa(ss);
      oa << state;
    }
    int size = ss.str().size()+1;

  // printf("%s: sending state to %d tag %d\n",__FUNCTION__, dest, state.GetTag());
    // MPI_Send( &size, 1, MPI_INT, i,
     // StateUniversal::tag, stateLocal.intercomm );
    if (request)
    {
      MPI_Isend( const_cast<char*>( ss.str().c_str() ), size, type, dest,
        state.GetTag(), comm, request);
    } else
    {
      MPI_Send( const_cast<char*>( ss.str().c_str() ), size, type, dest,
        state.GetTag(), comm );
    }
  }


  struct NetBuffer
  {
    NetBuffer()
     // : oa(ss), ia(ss)
    {

    }
  // std::vector<char> _buffer;
    size_t GetSize() { return ss.str().size()+1; }
    void* GetDataPtr() { return (void*)ss.str().c_str(); }
    std::stringstream ss;
  // boost::archive::text_oarchive oa;
  // boost::archive::text_iarchive ia;
  };

  class NetIBuffer :  public boost::archive::text_iarchive_impl<NetIBuffer>, public NetBuffer
  {
    NetIBuffer()
    : boost::archive::text_iarchive_impl<NetIBuffer>(ss,0)
    {}
    friend class boost::archive::detail::interface_iarchive<NetIBuffer>;
    friend class boost::archive::basic_text_iarchive<NetIBuffer>;
    friend class load_access;
    // boost::archive::text_iarchive ia;
  };

  struct State 
  {
  public:
    State()
    {
    // tag = 0;   
    }
    virtual int GetTag()
    {
      return -1;
    }

    virtual size_t Size() const
    {
      return _size;
    }
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
      assert(0);
    }

    virtual void Recv(int source, MPI_Comm comm, MPIBuffer& buffer)
    {
      assert(0);
    }
    virtual void Send(int dest, MPI_Comm comm)
    {
      assert(0);
    }
    virtual void Isend(int dest, MPI_Comm comm)
    {
      assert(0);
    }
    virtual void Irecv(int source, MPI_Comm comm, MPIBuffer& buffer)
    {
    // MPI_Iprobe(source, GetTag(), comm, &status, &request);
      assert(0);
    }
    virtual void Wait()
    {
      MPI_Wait(&request, &status);
    }
    virtual void SendAll(MPI_Comm comm)
    {
      int rank,size;
      MPI_Comm_rank(comm  , &rank);
      MPI_Comm_size(comm, &size);
      for(int i =0; i < size;i++)
      {
        if (i!= rank)
          Send(i,comm);
      }
    }
  // std::vector<char> _packBuffer;
    MPI_Status status;
    size_t _size;
    MPI_Request request;
  };

// NetBuffer& operator<<(NetBuffer& buffer, State& state)
// {
//   // buffer._buffer.resize(buffer._buffer.size()+state.GetPackBufferSize());
//   state.Pack(buffer._buffer);
//   return buffer;
// }
// NetBuffer& operator>>(NetBuffer& buffer, State& state)
// {
//   state.UnPack(buffer._buffer);
//   return buffer;
// }

  struct StateLocal : public State
  {
    MPI_Comm intercomm;
    int rank;
    int size;
    static int tag;
    virtual int GetTag()
    {
      return tag;
    }
    virtual void Recv(int source, MPI_Comm comm, MPIBuffer& buffer)
    {
      assert(0);
    }
    virtual void Send(int dest, MPI_Comm comm)
    {
      assert(0);
    }
  } ;

  struct StateMsg : public State
  {
    StateMsg()
    {
      _size = sizeof(this);
    }
    StateMsg(std::string msg_)
    :msg(msg_)
    {
      _size = sizeof(this);
    }
    virtual int GetTag()
    {
      return tag;
    }
    virtual void Recv(int source, MPI_Comm comm, MPIBuffer& buffer)
    {
      GetMPIState<StateMsg>(*this, comm, source, MPI_CHAR, buffer,status);
    }
    virtual void Send(int dest, MPI_Comm comm)
    {
      SetMPIState<StateMsg>(*this, comm, dest, MPI_CHAR);
    }

    std::string msg;
    static int tag;

    friend class boost::serialization::access;
    // When the class Archive corresponds to an output archive, the
    // & operator is defined similar to <<.  Likewise, when the class Archive
    // is a type of input archive the & operator is defined similar to >>.
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
      ar & msg;
    }
  } ;

  static int GVT_WORK_REQUEST;

  struct StateRequest : public State
  {
    StateRequest()
    {
      _size = sizeof(this);
      tagr=-1;
    }
    StateRequest(int tagr_)
    :tagr(tagr_)
    {
      _size = sizeof(this);
    }
    virtual int GetTag()
    {
      return tag;
    }
    void SetRequestedTag(int tag_)
    {
      tagr=tag_;
    }
    virtual void Recv(int source, MPI_Comm comm, MPIBuffer& buffer)
    {
      GetMPIState<StateRequest>(*this, comm, source, MPI_CHAR, buffer,status);
    }
    virtual void Send(int dest, MPI_Comm comm)
    {
      SetMPIState<StateRequest>(*this, comm, dest, MPI_CHAR);
    }

    int tagr;
    static int tag;

    friend class boost::serialization::access;
    // When the class Archive corresponds to an output archive, the
    // & operator is defined similar to <<.  Likewise, when the class Archive
    // is a type of input archive the & operator is defined similar to >>.
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
      ar & tagr;
    }
  } ;



  struct StateUniversal : public State
  {
    StateUniversal()
    :idDisplay(-1)
    {
      _size = sizeof(this);
    }
    virtual int GetTag()
    {
      return tag;
    }
    virtual void Recv(int source, MPI_Comm comm, MPIBuffer& buffer)
    {
      GetMPIState<StateUniversal>(*this, comm, source, MPI_CHAR, buffer,status);
    }
    virtual void Send(int dest, MPI_Comm comm)
    {
      SetMPIState<StateUniversal>(*this, comm, dest, MPI_CHAR);
    }

    int idDisplay;
    static int tag;

    friend class boost::serialization::access;
    // When the class Archive corresponds to an output archive, the
    // & operator is defined similar to <<.  Likewise, when the class Archive
    // is a type of input archive the & operator is defined similar to >>.
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
      ar & idDisplay;
    }
  } ;
// NetBuffer& operator<<(NetBuffer& buffer, StateUniversal& state)
// {
//    state.Pack(buffer._buffer);
//    return buffer;
// }

  struct uchar3
  {
    unsigned char r,g,b;
  };

template<class T>
  struct Framebuffer 
  {
    Framebuffer()
    :width(0),height(0),data(0)
    {}
    Framebuffer(int width, int height)
    {
      this->width = this->height = 0;
      Resize(width,height);
    }
    void Resize(int width, int height)
    {
      if (this->width == width && this->height == height)
        return;
      this->width = width;
      this->height = height;
    // data.resize(width*height);
      if (data)
        delete data;
      data = (T*)malloc(width*height*sizeof(T));
    }
    T* operator()(const int x,const int y)
    { return &data[x+width*y];}
    T* GetData()
    { return &data[0]; }

  // std::vector<char> data
    int width, height;
  // vector<T> data;
    T* data;
  };

  typedef Framebuffer<uchar3> FramebufferT;

template<class T>
  struct StatePixels : public State
  {
  // typedef PixelT ;
    StatePixels()
    {
      tag = 1524;
    // begin = 0;
    // end = 1;
      x = y = width = height = 0;
      framebuffer = NULL;
    }
  //end should be 1 past last element
  // StatePixels(size_t begin_, size_t end_)
  // : begin(begin_), end(end_)
    StatePixels(int x_, int y_, int width_, int height_, Framebuffer<T>* fb)
    : x(x_),y(y_),width(width_),height(height_), framebuffer(fb)
    {
      tag = 1524;
    // data.resize(width*height);
    }
    virtual int GetTag()
    {
      return tag;
    }
    virtual void Recv(int source, MPI_Comm comm, MPIBuffer& buffer)
    {
    // GetMPIState<StatePixels>(*this, comm, source, MPI_CHAR, buffer);
    // if (buffer.size() < 102)
      // buffer.resize(102);
      int info[5] = {0,0,0,0,1677};
      MPI_Recv(info, 5, MPI_INT, source, tag, comm, &status);
      x=info[0];
      y=info[1];
      width=info[2];
      height=info[3];
      unsigned int ptag = info[4];
     // ptag = info[4];
     // printf("statepixels recieving %d %d %d %d\n",x,y,width,height);
     // MPI_Probe(source, tag, comm, &status);

    // When probe returns, the status object has the size and other
    // attributes of the incoming message. Get the size of the message
  // MPI_Get_count(&status, type, &size); 
     // if (buffer.size() < info[2])
      // buffer.resize(info[2]);
    // printf("recv framebuffer size: %d %d\n", framebuffer->width, framebuffer->height);
      for(int y2=0;y2<height;y2++)
      {
       MPI_Recv((char*)(*framebuffer)(x,y+y2), width*sizeof(T), MPI_CHAR, source, ptag, comm, &status);
     }
   }
   virtual void Send(int dest, MPI_Comm comm)
   {
    // SetMPIState<StatePixels>(*this, comm, dest, MPI_CHAR);
    int ptag = 1677+tagOffset;
    int info[] = {x,y,width,height,ptag};
    MPI_Send(info, 5, MPI_INT, dest,
      tag, comm);
       // printf("send framebuffer size: %d %d\n", framebuffer->width, framebuffer->height);
  // printf("StatePixels::Send tag: %d\n", ptag);

    for(int y2=0;y2<height;y2++)
    {
     MPI_Send((*framebuffer)(x,y+y2), width*sizeof(T), MPI_CHAR, dest, ptag, comm);
   }
 }
 virtual void Isend(int dest, MPI_Comm comm)
 {
    // SetMPIState<StatePixels>(*this, comm, dest, MPI_CHAR, &request);
  for(int y=0;y<height;y++)
  {
    requests.push_back(MPI_Request());
    MPI_Request& request = requests.back();
       // MPI_Isend(const_cast<char*>(framebuffer(x+y)), width, MPI_CHAR, dest,
        // tag, comm, &request);
  }
}
virtual void Irecv(int dest, MPI_Comm comm, MPIBuffer& buffer)
{
  assert(0);
} 
virtual void Wait()
{
  for(int i =0;i<requests.size();i++)
  {
    MPI_Wait(&requests[i], &status);
  }
}
  //StatePixels(unsigned int begin, unsigned int end, void* data)
  //{}
friend class boost::serialization::access;
    template<class Archive>
void serialize(Archive & ar, const unsigned int version)
{
    // ar & begin;
    // ar & end;
  ar & x;
  ar & y;
  ar & width;
  ar & height;
    // ar & data;
}
  // size_t begin, end;
int x,y,width,height;
unsigned int tagOffset;
  // std::vector<char> data;
  // T* data;
Framebuffer<T>* framebuffer;
int tag;
std::vector<MPI_Request> requests;
};

typedef StatePixels<uchar3> StatePixelsT;

class Camera 
{
public:
  Camera(){}
  void Update()
  {
    direction = lookat-eye;
    float z = direction.length();
    direction= normalize(direction);
    up = glm::normalize(up);
    u = glm::cross(up, direction);
    v = glm::cross(direction,u);
    height = z*tan((3.14/180.0)*0.5*vfov);
    width = height*(vfov/hfov);
    u*=height;
    v*=width;
  }
  glm::vec3 eye, lookat, up, direction;
  float vfov,hfov;

  //protected:
  glm::vec3 u, v;
  float height, width;
};

struct StateFrame : public State
{
  StateFrame()
  {
    frame = 0;
  }
  virtual int GetTag()
  {
    return tag;
  }
  virtual void Recv(int source, MPI_Comm comm, MPIBuffer& buffer)
  {
    GetMPIState<StateFrame>(*this, comm, source, MPI_CHAR, buffer,status);
  }
  virtual void Send(int dest, MPI_Comm comm)
  {
    SetMPIState<StateFrame>(*this, comm, dest, MPI_CHAR);
  }
  // std::vector<char> data;
  int frame;
  short width, height;
  Camera camera;
  static int _frameCounter;
  static int tag;

  friend class boost::serialization::access;
    template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & frame;
    ar & width;
    ar & height;
    ar & camera.eye[0];
    ar & camera.eye[1];
    ar & camera.eye[2];
    ar & camera.lookat[0];
    ar & camera.lookat[1];
    ar & camera.lookat[2];
    ar & camera.up[0];
    ar & camera.up[1];
    ar & camera.up[2];
    ar & camera.hfov;
    ar & camera.vfov;
    
  }
};

struct StateDomain : public State
{
  StateDomain()
  {
  }
  virtual int GetTag()
  {
    return tag;
  }
  virtual void Recv(int source, MPI_Comm comm, MPIBuffer& buffer)
  {
    GetMPIState<StateDomain>(*this, comm, source, MPI_CHAR, buffer,status);
  }
  virtual void Send(int dest, MPI_Comm comm)
  {
    SetMPIState<StateDomain>(*this, comm, dest, MPI_CHAR);
  }
  // std::vector<char> data;
  glm::vec3 bound_min, bound_max;
  unsigned int id;
  virtual void Intersect() {}
  static int tag;

  friend class boost::serialization::access;
    template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & bound_min[0];
    ar & bound_min[1];
    ar & bound_min[2];
    ar & bound_max[0];
    ar & bound_max[1];
    ar & bound_max[2];
    ar & id;
  }
};

struct StateScene : public State
{
  StateScene()
  {
  }
  virtual int GetTag()
  {
    return tag;
  }
  virtual void Recv(int source, MPI_Comm comm, MPIBuffer& buffer)
  {
    GetMPIState<StateScene>(*this, comm, source, MPI_CHAR, buffer,status);
  }
  virtual void Send(int dest, MPI_Comm comm)
  {
    SetMPIState<StateScene>(*this, comm, dest, MPI_CHAR);
  }
  // std::vector<char> data;
  std::vector<StateDomain> domains;
  static int tag;

  friend class boost::serialization::access;
    template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & domains;
  }
};

struct gvtContext
{
  int mpi_rank;
  int mpi_size;
  // Camera* camera;
  // StateScene* scene;
};

struct StateWork : public State
{
  virtual void Run(const gvtContext& context)=0;
};

struct StateTile : public StateWork
{
  StateTile()
  : x(0), y(0), width(0), height(0), framebuffer(0)
  {
  }
  StateTile(int x_, int y_, int width_, int height_, Framebuffer<uchar3>* framebuffer=NULL)
  : x(x_),y(y_),width(width_),height(height_), framebuffer(framebuffer)
  {
  }
  virtual int GetTag()
  {
    return tag;
  }

  virtual void Recv(int source, MPI_Comm comm, MPIBuffer& buffer)
  {
    GetMPIState<StateTile>(*this, comm, source, MPI_CHAR, buffer,status);
  }
  virtual void Send(int dest, MPI_Comm comm)
  {
    SetMPIState<StateTile>(*this, comm, dest, MPI_CHAR);
  }
  virtual void Run(const gvtContext& context)
  {
    // 
    //  render
    //

    //
    // Send pixels
    //

    static StatePixelsT pixels;// = new StatePixels(0,0,102,102);
    static int psize = 0;
    // printf("TILE SIZE: %d %d\n", tile.width, tile.height);

    pixels.width= width;
    pixels.height=height;
    pixels.framebuffer = framebuffer;
    pixels.x = x;
    pixels.y = y;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    pixels.tagOffset = rank*1000;
    // printf("tagOFfset: %d\n", tagOffset);

    pixels.Send(0, MPI_COMM_WORLD);

  // for(size_t i =0;i < pixels_sent.size();i++)
  // {
  //   // pixels_sent[i]->Wait();
  //   // delete pixels_sent[i];
  // }
  // pixels_sent_c = 0;

  // boost::timer::cpu_times const elapsed_times(render_timer.elapsed());
       // boost::timer::nanosecond_type const elapsed(elapsed_times.system
        // + elapsed_times.user);
       // cout << boost::timer::format(render_timer.elapsed(), 3, "total render: %w seconds\n");

  // printf("total render time %d: %f s\n", rank, double(render_times_accumulated/1000LL)/1000000.0d);
  // printf("total pixel time %d: %f s\n", rank, double(pixel_times_accumulated/1000LL)/1000000.0d);

  }

  unsigned int x,y,width,height;
  Framebuffer<uchar3>* framebuffer;
  static int tag;

  friend class boost::serialization::access;
    template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & x;
    ar & y;
    ar & width;
    ar & height;
  }
};

}

#endif
