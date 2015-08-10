<h1>GraviT: A Comprehensive Ray Tracing Framework for Visualization in Distributed-Memory Parallel Environments </h1>

<a href="http://tacc.github.io/GraviT/">http://tacc.github.io/GraviT/</a>

<h2>Overview</h2>

GraviT is a software library for the class of simulation problems where insight is derived from actors operating on scientific data, i.e., data that has physical coordinates.  This data is often so large that it cannot reside in the memory of a single compute node.  While GraviT is designed with many types of actors and use cases in mind, the canonical usage of GraviT is with the actors that are rays and data that are tessellated surfaces.  In this case, GraviT produces ray-traced renderings.

GraviT’s design focuses on three key elements: interface, scheduler, and engine.   The interface element is how users interact with GraviT.  The scheduler element focuses on how to bring together actors with the appropriate pieces of data to advance the calculation.  The engine element performs the specified operations of the actor upon the data. This design is intentionally modular: developers can opt to extend GraviT with their own implementations of interface, scheduler, or engine, and to re-use the implementations from the other areas.   In short, GraviT provides a fully working system, but also one that can be easily extended.  Finally, GraviT is intended for very computationally heavy problems, so it aims to carry out calculations in the most efficient way possible while maintaining modularity and generality.  This goal impacts the scheduler and engine elements in particular.  

GraviT is divided into “core” infrastructure and domain-specific library.  The “core” infrastructure, abbreviated GVT-Core, contains abstract types, as well as implementations that are common to domain-specific libraries, for example scheduling.  The domain-specific libraries build on GVT-Core to create a functional system that is specialized to their area.  The domain-specific libraries that have been discussed by the GraviT team are:
<ul>
<li>GVT-Render, for ray-tracing geometric surfaces</li>
<li>GVT-Volume, for volume rendering</li>
<li>GVT-Advect, for particle advection</li>
<li>GVT-Photo, for modeling photon interactions</li>
</ul>

<h2>Engines</h2>

GraviT’s engines are the modules that carry out the calculation to advance an actor using information from the appropriate data.  The choice of the word “engine” conveys a connotation that this operation is computationally intensive, and this is purposeful.  GraviT is aimed at problems where determining the behavior of an actor is time-intensive because (1) there are many actors, (2) the data is very large, or (3) both.  Building high-quality engines takes significant development time.  Fortunately, existing third-party products can fill this role in key instances.  GraviT’s development strategy is inclusive of these third-party products, and provides options for leveraging them.  For these cases, the GraviT engine acts as an adaptor, converting requests from scheduler into instructions that the third-party product can accept.  At this time, the third-party products most considered by the GraviT team are Intel’s Embree/OSPRay and NVIDIA’s Optix Prime/Optix.  These libraries are being used for GVT-Render, but also have been discussed for GVT-Volume and GVT-Advect.  Of course, a GraviT engine does not need to be an adaptor around a third-party product.  Native GraviT engines are also supported, i.e., engines that take instructions from the schedules and carry them out directly.   An example of a native engine is the ray tracer written specifically for GraviT, using data-parallel primitives in the EAVL framework.

<h2>Interfaces</h2>

GraviT’s interface is how external applications interact with GraviT.  GraviT has a single interface, and all domain-specific applications use this interface.  The interface is key-based, which means that each must specify its set of supported keys, and document that list for library users.  (For example, GVT-Render will support the notion of a camera position, and its documentation must make clear that the key associated with this position is “Camera”, as opposed to “CamPosition”.)  One motivation behind this design is the code for the interface can be written once, and re-used for each domain-specific application, modulo defining the sets of keys that are used.   

Beyond the general interface, new interfaces to GraviT can added as wrappers around the main interface.  These interfaces will define their own library functions, and implement the functions by translating the incoming information to the main GraviT interface.  One reason to define a new, wrapper interface is to simplify the interface for users, i.e., to reduce barriers to community usage.  Another reason is to fit a legacy interface, once again to reduce barriers to community usage.  An example of this latter type exists in GVT-Render, which provides a GLuRay interface.  The GLuRay interface meets the OpenGL 1.x standards, which is used by multiple visualization tools, and thus allows them to use GraviT with minimal overhead.  The GLuRay wrapper interface then implements its functions by calling functions in the main GraviT interface. 

GraviT’s interface also sometimes needs to affect application behavior.  This is done through callbacks.  Applications can register callbacks to load or unload data with GraviT, and GraviT’s schedule can issue these callbacks while it is executing.  

<h2>Data Model</h2>

GraviT’s data model varies from domain-specific library to domain-specific library.  The application callbacks to load data need to be aware of the data model of the domain-specific library to load it into GraviT’s form.  Our team has discussed issues such as “zero-copy” in situ, but this discussion has not yet led to a fixed model.  As this issue is large in scope, our short-term plan appears to be that we make a copy of the data in GraviT’s format.

GraviT does not only accept data through load and unload callbacks.  Applications can specify the data before execution starts.  This is the case with GLuRay, where data is acquired incrementally through its interface, and then registered with the scheduler.
Scheduler 

The job of GraviT’s scheduler is to get actor and data together so that an engine can carry out its operations.  When GraviT runs in a distributed-memory parallel setting, getting the right actor and data together can have significant latency.  There is a spectrum of algorithms that respond to this issue.  On one extreme, actors stay on the same node throughout execution, and the needed data is imported to carry out execution.  On the other extreme, data stays on the same node throughout execution, and actors are passed around between nodes.

Our goal with the GraviT schedulers is that we develop building blocks to facilitate rapid development of new schedulers.  We endeavor to have schedulers that take approximately forty lines of code, and who’s implementations highly resemble the pseudocode we would use to describe them in a publication. 

GraviT does make assumptions about the data it operates on.  Specifically, it assumes that data that is very large can be decomposed into domains, i.e., partitioning the set of all data so that each domain is spatially contiguous.  Further, GraviT’s schedulers assume that they can access information about: (1) the spatial extents of each domain and (2) the cost of acquiring each domain.  These requirements are important for enabling schedulers to perform efficiently.   
