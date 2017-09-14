//
// Created by Joao Barbosa on 9/13/17.
//

#include "rcontext.h"


template<> cntx::node cntx::node::error_node = cntx::node();
template<> std::shared_ptr< cntx::context<cntx::Variant, cntx::rcontext> > cntx::context<cntx::Variant, cntx::rcontext>::_singleton = nullptr;