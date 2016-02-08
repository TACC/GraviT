/*
 * Copyright (C) 2007,  The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from U.S. Department of Energy).
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by the Visualization
 *      Group at Lawrence Berkeley National Laboratory.
 * 4. Neither the name of the University of California, Berkeley nor of the 
 *    Lawrence Berkeley National Laboratory may be used to endorse or 
 *    promote products derived from this software without specific prior 
 *    written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * You are under no obligation whatsoever to provide any bug fixes, patches, 
 * or upgrades to the features, functionality or performance of the source 
 * code ("Enhancements") to anyone; however, if you choose to make your 
 * Enhancements available either publicly, or directly to Lawrence Berkeley 
 * National Laboratory, without imposing a separate written license agreement 
 * for such Enhancements, then you hereby grant the following license: a 
 * non-exclusive, royalty-free perpetual license to install, use, modify, 
 * prepare derivative works, incorporate into other computer software, 
 * distribute, and sublicense such Enhancements or derivative works thereof, 
 * in binary and source code form. 
 *
 * This work is supported by the U. S. Department of Energy under contract 
 * number DE-AC03-76SF00098 between the U. S. Department of Energy and the 
 * University of California.
 *
 *	Author: Wes Bethel
 *		Lawrence Berkeley National Laboratory
 *              Berkeley, California 
 *
 *  "this software is 100% hand-crafted by a human being in the USA"
 */

/*
 *  $Id: timer.h,v 1.1.1.1 2007/03/27 16:03:56 wes Exp $
 *  Version: $Name: May_21_2007 $
 *  $Revision: 1.1.1.1 $
 *  $Log: timer.h,v $
 *  Revision 1.1.1.1  2007/03/27 16:03:56  wes
 *  Initial checkin
 *
 */

#ifndef __timer_h
#define __timer_h

#include <sys/time.h>

typedef struct
{
    long sec;
    long usec;
} my_timer_t;

#ifdef __cplusplus
extern "C" {
#endif

int timeCurrent(my_timer_t *t);
double timeDifferenceMS(const my_timer_t *start, const my_timer_t *end);

#ifdef __cplusplus
}
#endif

#endif
/* EOF */
