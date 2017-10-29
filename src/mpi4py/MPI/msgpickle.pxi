# -----------------------------------------------------------------------------

cdef extern from "Python.h":
    bint       PyBytes_CheckExact(object)
    char*      PyBytes_AsString(object) except NULL
    Py_ssize_t PyBytes_Size(object) except -1
    object     PyBytes_FromStringAndSize(char*,Py_ssize_t)
    object     PyBytes_Join"_PyBytes_Join"(object,object)

# -----------------------------------------------------------------------------

cdef object PyPickle_dumps = None
cdef object PyPickle_loads = None
cdef object PyPickle_PROTOCOL = None
if PY3:
    from pickle import dumps as PyPickle_dumps
    from pickle import loads as PyPickle_loads
    from pickle import DEFAULT_PROTOCOL as PyPickle_PROTOCOL
else:
    try:
        from cPickle import dumps as PyPickle_dumps
        from cPickle import loads as PyPickle_loads
        from cPickle import HIGHEST_PROTOCOL as PyPickle_PROTOCOL
    except ImportError:
        from pickle  import dumps as PyPickle_dumps
        from pickle  import loads as PyPickle_loads
        from pickle  import HIGHEST_PROTOCOL as PyPickle_PROTOCOL

cdef object PyBytesIO_New = None
cdef object PyPickle_loadf = None
if PY2:
    try:
        from cStringIO import StringIO as PyBytesIO_New
    except ImportError:
        from io import BytesIO as PyBytesIO_New
    try:
        from cPickle import load as PyPickle_loadf
    except ImportError:
        from pickle import load as PyPickle_loadf

@cython.final
@cython.internal
cdef class Pickle:

    """
    Pickle/unpickle Python objects
    """

    cdef object ob_dumps
    cdef object ob_loads
    cdef object ob_PROTO

    def __cinit__(self, *args, **kwargs):
        self.ob_dumps = PyPickle_dumps
        self.ob_loads = PyPickle_loads
        self.ob_PROTO = PyPickle_PROTOCOL

    def __init__(self, dumps=None, loads=None, protocol=None):
        if dumps is None:
            dumps = PyPickle_dumps
        if loads is None:
            loads = PyPickle_loads
        if protocol is None:
            if dumps is PyPickle_dumps:
                protocol = PyPickle_PROTOCOL
        self.ob_dumps = dumps
        self.ob_loads = loads
        self.ob_PROTO = protocol

    cpdef object dumps(self, object obj):
        "dumps(obj) -> bytes"
        if self.ob_PROTO is not None:
            return self.ob_dumps(obj, self.ob_PROTO)
        else:
            return self.ob_dumps(obj)

    cpdef object loads(self, object buf):
        "loads(buf) -> object"
        if PY2 and not PyBytes_CheckExact(buf):
            if self.ob_loads is PyPickle_loads:
                buf = PyBytesIO_New(buf)
                return PyPickle_loadf(buf)
        return self.ob_loads(buf)

    property PROTOCOL:
        "protocol"
        def __get__(self):
            return self.ob_PROTO
        def __set__(self, protocol):
            if protocol is None:
                if self.ob_dumps is PyPickle_dumps:
                    protocol = PyPickle_PROTOCOL
            self.ob_PROTO = protocol

    cdef object dump(self, object obj, void **p, int *n):
        if obj is None:
            p[0] = NULL
            n[0] = 0
            return None
        cdef object buf = self.dumps(obj)
        p[0] = <void*>PyBytes_AsString(buf)
        n[0] = downcast(PyBytes_Size(buf))
        return buf

    cdef object load(self, void *p, int n):
        if p == NULL or n == 0: return None
        return self.loads(tobuffer(p, n))

    cdef object dumpv(self, object obj, void **p, int n, int cnt[], int dsp[]):
        cdef Py_ssize_t i=0, m=n
        if obj is None:
            p[0] = NULL
            for i from 0 <= i < m:
                cnt[i] = 0
                dsp[i] = 0
            return None
        cdef object items = list(obj)
        m = len(items)
        if m != n: raise ValueError(
            "expecting %d items, got %d" % (n, m))
        cdef int d=0, c=0
        for i from 0 <= i < m:
            items[i] = self.dump(items[i], p, &c)
            if c == 0: items[i] = b''
            cnt[i] = c; dsp[i] = d
            d = downcast(<MPI_Aint>d + <MPI_Aint>c)
        cdef object buf = PyBytes_Join(b'', items)
        p[0] = PyBytes_AsString(buf)
        return buf

    cdef object loadv(self, void *p, int n, int cnt[], int dsp[]):
        cdef Py_ssize_t i=0, m=n
        cdef object items = [None] * m
        if p == NULL: return items
        for i from 0 <= i < m:
            items[i] = self.load(<char*>p+dsp[i], cnt[i])
        return items

    cdef object alloc(self, void **p, int n):
        if n == 0:
            p[0] = NULL
            return None
        cdef object buf = PyBytes_FromStringAndSize(NULL, n)
        p[0] = PyBytes_AsString(buf)
        return buf

    cdef object allocv(self, void **p, int n, int cnt[], int dsp[]):
        cdef int i=0, d=0
        for i from 0 <= i < n:
            dsp[i] = d
            d += cnt[i]
        return self.alloc(p, d)


cdef Pickle PyMPI_PICKLE = Pickle()
pickle = PyMPI_PICKLE

# -----------------------------------------------------------------------------

cdef inline object allocate_count_displ(int n, int **p, int **q):
    cdef object mem = allocate(2*n, sizeof(int), p)
    q[0] = p[0] + n
    return mem

# -----------------------------------------------------------------------------

cdef object PyMPI_send(object obj, int dest, int tag,
                       MPI_Comm comm):
    cdef Pickle pickle = PyMPI_PICKLE
    #
    cdef void *sbuf = NULL
    cdef int scount = 0
    cdef MPI_Datatype stype = MPI_BYTE
    #
    cdef object tmps = None
    if dest != MPI_PROC_NULL:
        tmps = pickle.dump(obj, &sbuf, &scount)
    with nogil: CHKERR( MPI_Send(sbuf, scount, stype,
                                 dest, tag, comm) )
    return None


cdef object PyMPI_bsend(object obj, int dest, int tag,
                        MPI_Comm comm):
    cdef Pickle pickle = PyMPI_PICKLE
    #
    cdef void *sbuf = NULL
    cdef int scount = 0
    cdef MPI_Datatype stype = MPI_BYTE
    #
    cdef object tmps = None
    if dest != MPI_PROC_NULL:
        tmps = pickle.dump(obj, &sbuf, &scount)
    with nogil: CHKERR( MPI_Bsend(sbuf, scount, stype,
                                  dest, tag, comm) )
    return None


cdef object PyMPI_ssend(object obj, int dest, int tag,
                        MPI_Comm comm):
    cdef Pickle pickle = PyMPI_PICKLE
    #
    cdef void *sbuf = NULL
    cdef int scount = 0
    cdef MPI_Datatype stype = MPI_BYTE
    #
    cdef object tmps = None
    if dest != MPI_PROC_NULL:
        tmps = pickle.dump(obj, &sbuf, &scount)
    with nogil: CHKERR( MPI_Ssend(sbuf, scount, stype,
                                  dest, tag, comm) )
    return None

# -----------------------------------------------------------------------------

cdef extern from "Python.h":
    int PyErr_WarnEx(object, char*, int) except -1

cdef object PyMPI_recv_obarg(object obj, int source, int tag,
                             MPI_Comm comm, MPI_Status *status):
    cdef Pickle pickle = PyMPI_PICKLE
    #
    cdef void *rbuf = NULL
    cdef int rcount = 0
    cdef MPI_Datatype rtype = MPI_BYTE
    cdef MPI_Status rsts
    cdef object   rmsg = None
    cdef MPI_Aint rlen = 0
    #
    PyErr_WarnEx(UserWarning, b"the 'buf' argument is deprecated", 1)
    #
    if source != MPI_PROC_NULL:
        if is_integral(obj):
            rcount = <int> obj
            rmsg = pickle.alloc(&rbuf, rcount)
        else:
            rmsg = getbuffer_w(obj, &rbuf, &rlen)
            rcount = clipcount(rlen)
        if status == MPI_STATUS_IGNORE:
            status = &rsts
    with nogil:
        CHKERR( MPI_Recv(rbuf, rcount, rtype,
                         source, tag, comm, status) )
        if source != MPI_PROC_NULL:
            CHKERR( MPI_Get_count(status, rtype, &rcount) )
    #
    if rcount <= 0: return None
    return pickle.load(rbuf, rcount)


cdef object PyMPI_recv_match(object obj, int source, int tag,
                             MPI_Comm comm, MPI_Status *status):
    cdef Pickle pickle = PyMPI_PICKLE
    #
    cdef void *rbuf = NULL
    cdef int rcount = 0
    cdef MPI_Datatype rtype = MPI_BYTE
    #
    cdef MPI_Message match = MPI_MESSAGE_NULL
    cdef MPI_Status rsts
    cdef object rmsg = None
    <void>obj # unused
    #
    with nogil:
        CHKERR( MPI_Mprobe(source, tag, comm, &match, &rsts) )
        CHKERR( MPI_Get_count(&rsts, rtype, &rcount) )
    rmsg = pickle.alloc(&rbuf, rcount)
    with nogil:
        CHKERR( MPI_Mrecv(rbuf, rcount, rtype, &match, status) )
    #
    if rcount <= 0: return None
    return pickle.load(rbuf, rcount)


cdef object PyMPI_recv_probe(object obj, int source, int tag,
                             MPI_Comm comm, MPI_Status *status):
    cdef Pickle pickle = PyMPI_PICKLE
    #
    cdef void *rbuf = NULL
    cdef int rcount = 0
    cdef MPI_Datatype rtype = MPI_BYTE
    #
    cdef MPI_Status rsts
    cdef object rmsg = None
    <void>obj # unused
    #
    with PyMPI_Lock(comm, "recv"):
        with nogil:
            CHKERR( MPI_Probe(source, tag, comm, &rsts) )
            CHKERR( MPI_Get_count(&rsts, rtype, &rcount) )
            source = rsts.MPI_SOURCE; tag = rsts.MPI_TAG
        rmsg = pickle.alloc(&rbuf, rcount)
        with nogil:
            CHKERR( MPI_Recv(rbuf, rcount, rtype,
                             source, tag, comm, status) )
    #
    if rcount <= 0: return None
    return pickle.load(rbuf, rcount)


cdef object PyMPI_recv(object obj, int source, int tag,
                       MPI_Comm comm, MPI_Status *status):
    if obj is not None:
        return PyMPI_recv_obarg(obj, source, tag, comm, status)
    elif options.recv_mprobe:
        return PyMPI_recv_match(obj, source, tag, comm, status)
    else:
        return PyMPI_recv_probe(obj, source, tag, comm, status)

# -----------------------------------------------------------------------------

cdef object PyMPI_isend(object obj, int dest, int tag,
                        MPI_Comm comm, MPI_Request *request):
    cdef Pickle pickle = PyMPI_PICKLE
    #
    cdef void *sbuf = NULL
    cdef int scount = 0
    cdef MPI_Datatype stype = MPI_BYTE
    #
    cdef object smsg = None
    if dest != MPI_PROC_NULL:
        smsg = pickle.dump(obj, &sbuf, &scount)
    with nogil: CHKERR( MPI_Isend(sbuf, scount, stype,
                                  dest, tag, comm, request) )
    return smsg


cdef object PyMPI_ibsend(object obj, int dest, int tag,
                         MPI_Comm comm, MPI_Request *request):
    cdef Pickle pickle = PyMPI_PICKLE
    #
    cdef void *sbuf = NULL
    cdef int scount = 0
    cdef MPI_Datatype stype = MPI_BYTE
    #
    cdef object smsg = None
    if dest != MPI_PROC_NULL:
        smsg = pickle.dump(obj, &sbuf, &scount)
    with nogil: CHKERR( MPI_Ibsend(sbuf, scount, stype,
                                   dest, tag, comm, request) )
    return smsg


cdef object PyMPI_issend(object obj, int dest, int tag,
                         MPI_Comm comm, MPI_Request *request):
    cdef Pickle pickle = PyMPI_PICKLE
    #
    cdef void *sbuf = NULL
    cdef int scount = 0
    cdef MPI_Datatype stype = MPI_BYTE
    #
    cdef object smsg = None
    if dest != MPI_PROC_NULL:
        smsg = pickle.dump(obj, &sbuf, &scount)
    with nogil: CHKERR( MPI_Issend(sbuf, scount, stype,
                                   dest, tag, comm, request) )
    return smsg


cdef object PyMPI_irecv(object obj, int source, int tag,
                        MPI_Comm comm, MPI_Request *request):
    cdef Pickle pickle = PyMPI_PICKLE
    #
    cdef void *rbuf = NULL
    cdef MPI_Aint rlen = 0
    cdef int rcount = 0
    cdef MPI_Datatype rtype = MPI_BYTE
    #
    cdef object rmsg = None
    if source != MPI_PROC_NULL:
        if obj is None:
            rcount = <int>(1<<15)
            obj = pickle.alloc(&rbuf, rcount)
            rmsg = getbuffer_r(obj, NULL, NULL)
        elif is_integral(obj):
            rcount = <int> obj
            obj = pickle.alloc(&rbuf, rcount)
            rmsg = getbuffer_r(obj, NULL, NULL)
        else:
            rmsg = getbuffer_w(obj, &rbuf, &rlen)
            rcount = clipcount(rlen)
    with nogil: CHKERR( MPI_Irecv(rbuf, rcount, rtype,
                                  source, tag, comm, request) )
    return rmsg

# -----------------------------------------------------------------------------

cdef object PyMPI_sendrecv(object sobj, int dest,   int sendtag,
                           object robj, int source, int recvtag,
                           MPI_Comm comm, MPI_Status *status):
    cdef MPI_Request request = MPI_REQUEST_NULL
    sobj = PyMPI_isend(sobj, dest,   sendtag, comm, &request)
    robj = PyMPI_recv (robj, source, recvtag, comm, status)
    with nogil: CHKERR( MPI_Wait(&request, MPI_STATUS_IGNORE) )
    return robj

# -----------------------------------------------------------------------------

cdef object PyMPI_load(MPI_Status *status, object ob):
    cdef Pickle pickle = PyMPI_PICKLE
    cdef void *rbuf = NULL
    cdef int rcount = 0
    cdef MPI_Datatype rtype = MPI_BYTE
    if type(ob) is not memory: return None
    CHKERR( MPI_Get_count(status, rtype, &rcount) )
    if rcount <= 0: return None
    ob = asmemory(ob, &rbuf, NULL)
    return pickle.load(rbuf, rcount)


cdef object PyMPI_wait(Request request, Status status):
    cdef object buf
    #
    cdef MPI_Status rsts
    with nogil: CHKERR( MPI_Wait(&request.ob_mpi, &rsts) )
    buf = request.ob_buf
    if status is not None:
        status.ob_mpi = rsts
    if request.ob_mpi == MPI_REQUEST_NULL:
        request.ob_buf = None
    #
    return PyMPI_load(&rsts, buf)


cdef object PyMPI_test(Request request, int *flag, Status status):
    cdef object buf = None
    #
    cdef MPI_Status rsts
    with nogil: CHKERR( MPI_Test(&request.ob_mpi, flag, &rsts) )
    if flag[0]:
        buf = request.ob_buf
    if status is not None:
        status.ob_mpi = rsts
    if request.ob_mpi == MPI_REQUEST_NULL:
        request.ob_buf = None
    #
    if not flag[0]: return None
    return PyMPI_load(&rsts, buf)


cdef object PyMPI_waitany(requests, int *index, Status status):
    cdef object buf = None
    #
    cdef int count = 0
    cdef MPI_Request *irequests = NULL
    cdef MPI_Status rsts
    #
    cdef tmp = acquire_rs(requests, None, &count, &irequests, NULL)
    try:
        with nogil: CHKERR( MPI_Waitany(count, irequests, index, &rsts) )
        if index[0] != MPI_UNDEFINED:
            buf = (<Request>requests[index[0]]).ob_buf
        if status is not None:
            status.ob_mpi = rsts
    finally:
        release_rs(requests, None, count, irequests, NULL)
    #
    if index[0] == MPI_UNDEFINED: return None
    return PyMPI_load(&rsts, buf)


cdef object PyMPI_testany(requests, int *index, int *flag, Status status):
    cdef object buf = None
    #
    cdef int count = 0
    cdef MPI_Request *irequests = NULL
    cdef MPI_Status rsts
    #
    cdef tmp = acquire_rs(requests, None, &count, &irequests, NULL)
    try:
        with nogil: CHKERR( MPI_Testany(count, irequests, index, flag, &rsts) )
        if index[0] != MPI_UNDEFINED:
            buf = (<Request>requests[index[0]]).ob_buf
        if status is not None:
            status.ob_mpi = rsts
    finally:
        release_rs(requests, None, count, irequests, NULL)
    #
    if index[0] == MPI_UNDEFINED: return None
    if not flag[0]: return None
    return PyMPI_load(&rsts, buf)


cdef object PyMPI_waitall(requests, statuses):
    cdef object bufs = None
    #
    cdef Py_ssize_t i = 0
    cdef int count = 0
    cdef MPI_Request *irequests = NULL
    cdef MPI_Status *istatuses = MPI_STATUSES_IGNORE
    #
    cdef tmp = acquire_rs(requests, True, &count, &irequests, &istatuses)
    try:
        with nogil: CHKERR( MPI_Waitall(count, irequests, istatuses) )
        bufs = [(<Request>requests[i]).ob_buf for i from 0 <= i < count]
    finally:
        release_rs(requests, statuses, count, irequests, istatuses)
    #
    return [PyMPI_load(&istatuses[i], bufs[i]) for i from 0 <= i < count]


cdef object PyMPI_testall(requests, int *flag, statuses):
    cdef object bufs = None
    #
    cdef Py_ssize_t i = 0
    cdef int count = 0
    cdef MPI_Request *irequests = NULL
    cdef MPI_Status *istatuses = MPI_STATUSES_IGNORE
    #
    cdef tmp = acquire_rs(requests, True, &count, &irequests, &istatuses)
    try:
        with nogil: CHKERR( MPI_Testall(count, irequests, flag, istatuses) )
        if flag[0]:
            bufs = [(<Request>requests[i]).ob_buf for i from 0 <= i < count]
    finally:
        release_rs(requests, statuses, count, irequests, istatuses)
    #
    if not flag[0]: return None
    return [PyMPI_load(&istatuses[i], bufs[i]) for i from 0 <= i < count]

# -----------------------------------------------------------------------------

cdef object PyMPI_probe(int source, int tag,
                        MPI_Comm comm, MPI_Status *status):
    with nogil: CHKERR( MPI_Probe(source, tag, comm, status) )
    return True

cdef object PyMPI_iprobe(int source, int tag,
                         MPI_Comm comm, MPI_Status *status):
    cdef int flag = 0
    with nogil: CHKERR( MPI_Iprobe(source, tag, comm, &flag, status) )
    return <bint>flag

cdef object PyMPI_mprobe(int source, int tag, MPI_Comm comm,
                         MPI_Message *message, MPI_Status *status):
    cdef Pickle pickle = PyMPI_PICKLE
    cdef void* rbuf = NULL
    cdef int rcount = 0
    cdef MPI_Datatype rtype = MPI_BYTE
    cdef MPI_Status rsts
    if (status == MPI_STATUS_IGNORE): status = &rsts
    with nogil: CHKERR( MPI_Mprobe(source, tag, comm, message, status) )
    if message[0] == MPI_MESSAGE_NO_PROC: return None
    CHKERR( MPI_Get_count(status, rtype, &rcount) )
    cdef object rmsg = pickle.alloc(&rbuf, rcount)
    return rmsg

cdef object PyMPI_improbe(int source, int tag, MPI_Comm comm, int *flag,
                          MPI_Message *message, MPI_Status *status):
    cdef Pickle pickle = PyMPI_PICKLE
    cdef void* rbuf = NULL
    cdef int rcount = 0
    cdef MPI_Datatype rtype = MPI_BYTE
    cdef MPI_Status rsts
    if (status == MPI_STATUS_IGNORE): status = &rsts
    with nogil: CHKERR( MPI_Improbe(source, tag, comm, flag, message, status) )
    if flag[0] == 0 or message[0] == MPI_MESSAGE_NO_PROC: return None
    CHKERR( MPI_Get_count(status, rtype, &rcount) )
    cdef object rmsg = pickle.alloc(&rbuf, rcount)
    return rmsg

cdef object PyMPI_mrecv(object rmsg,
                        MPI_Message *message, MPI_Status *status):
    cdef Pickle pickle = PyMPI_PICKLE
    cdef void* rbuf = NULL
    cdef MPI_Aint rlen = 0
    cdef MPI_Datatype rtype = MPI_BYTE
    if message[0] == MPI_MESSAGE_NO_PROC:
        rmsg = None
    elif rmsg is None:
        pass
    elif PyBytes_CheckExact(rmsg):
        rmsg = getbuffer_r(rmsg, &rbuf, &rlen)
    else:
        rmsg = getbuffer_w(rmsg, &rbuf, &rlen)
    cdef int rcount = clipcount(rlen)
    with nogil: CHKERR( MPI_Mrecv(rbuf, rcount, rtype, message, status) )
    rmsg = pickle.load(rbuf, rcount)
    return rmsg

cdef object PyMPI_imrecv(object rmsg,
                         MPI_Message *message, MPI_Request *request):
    cdef void* rbuf = NULL
    cdef MPI_Aint rlen = 0
    cdef MPI_Datatype rtype = MPI_BYTE
    if message[0] == MPI_MESSAGE_NO_PROC:
        rmsg = None
    elif rmsg is None:
        pass
    elif PyBytes_CheckExact(rmsg):
        rmsg = getbuffer_r(rmsg, &rbuf, &rlen)
    else:
        rmsg = getbuffer_w(rmsg, &rbuf, &rlen)
    cdef int rcount = clipcount(rlen)
    with nogil: CHKERR( MPI_Imrecv(rbuf, rcount, rtype, message, request) )
    return rmsg

# -----------------------------------------------------------------------------

cdef object PyMPI_barrier(MPI_Comm comm):
    with nogil: CHKERR( MPI_Barrier(comm) )
    return None


cdef object PyMPI_bcast(object obj, int root, MPI_Comm comm):
    cdef Pickle pickle = PyMPI_PICKLE
    #
    cdef void *buf = NULL
    cdef int count = 0
    cdef MPI_Datatype dtype = MPI_BYTE
    #
    cdef int dosend=0, dorecv=0
    cdef int inter=0, rank=0
    CHKERR( MPI_Comm_test_inter(comm, &inter) )
    if inter:
        if root == MPI_PROC_NULL:
            dosend=0; dorecv=0;
        elif root == MPI_ROOT:
            dosend=1; dorecv=0;
        else:
            dosend=0; dorecv=1;
    else:
        CHKERR( MPI_Comm_rank(comm, &rank) )
        if root == rank:
            dosend=1; dorecv=1;
        else:
            dosend=0; dorecv=1;
    #
    cdef object smsg = None
    cdef object rmsg = None
    #
    if dosend: smsg = pickle.dump(obj, &buf, &count)
    if dosend and dorecv: rmsg = smsg
    with PyMPI_Lock(comm, "bcast"):
        with nogil: CHKERR( MPI_Bcast(
            &count, 1, MPI_INT,
            root, comm) )
        if dorecv and not dosend:
            rmsg = pickle.alloc(&buf, count)
        with nogil: CHKERR( MPI_Bcast(
            buf, count, dtype,
            root, comm) )
    if dorecv: rmsg = pickle.load(buf, count)
    #
    return rmsg


cdef object PyMPI_gather(object sendobj, int root, MPI_Comm comm):
    cdef Pickle pickle = PyMPI_PICKLE
    #
    cdef void *sbuf = NULL
    cdef int scount = 0
    cdef MPI_Datatype stype = MPI_BYTE
    cdef void *rbuf = NULL
    cdef int *rcounts = NULL
    cdef int *rdispls = NULL
    cdef MPI_Datatype rtype = MPI_BYTE
    #
    cdef int dosend=0, dorecv=0
    cdef int inter=0, size=0, rank=0
    CHKERR( MPI_Comm_test_inter(comm, &inter) )
    if inter:
        CHKERR( MPI_Comm_remote_size(comm, &size) )
        if root == MPI_PROC_NULL:
            dosend=0; dorecv=0;
        elif root == MPI_ROOT:
            dosend=0; dorecv=1;
        else:
            dosend=1; dorecv=0;
    else:
        CHKERR( MPI_Comm_size(comm, &size) )
        CHKERR( MPI_Comm_rank(comm, &rank) )
        if root == rank:
            dosend=1; dorecv=1;
        else:
            dosend=1; dorecv=0;
    #
    cdef object tmps = None
    cdef object rmsg = None
    cdef object tmp1
    #
    if dorecv: tmp1 = allocate_count_displ(size, &rcounts, &rdispls)
    if dosend: tmps = pickle.dump(sendobj, &sbuf, &scount)
    with PyMPI_Lock(comm, "gather"):
        with nogil: CHKERR( MPI_Gather(
            &scount, 1, MPI_INT,
            rcounts, 1, MPI_INT,
            root, comm) )
        if dorecv: rmsg = pickle.allocv(&rbuf, size, rcounts, rdispls)
        with nogil: CHKERR( MPI_Gatherv(
            sbuf, scount,           stype,
            rbuf, rcounts, rdispls, rtype,
            root, comm) )
    if dorecv: rmsg = pickle.loadv(rbuf, size, rcounts, rdispls)
    #
    return rmsg


cdef object PyMPI_scatter(object sendobj, int root, MPI_Comm comm):
    cdef Pickle pickle = PyMPI_PICKLE
    #
    cdef void *sbuf = NULL
    cdef int *scounts = NULL
    cdef int *sdispls = NULL
    cdef MPI_Datatype stype = MPI_BYTE
    cdef void *rbuf = NULL
    cdef int rcount = 0
    cdef MPI_Datatype rtype = MPI_BYTE
    #
    cdef int dosend=0, dorecv=0
    cdef int inter=0, size=0, rank=0
    CHKERR( MPI_Comm_test_inter(comm, &inter) )
    if inter:
        CHKERR( MPI_Comm_remote_size(comm, &size) )
        if root == MPI_PROC_NULL:
            dosend=0; dorecv=0;
        elif root == MPI_ROOT:
            dosend=1; dorecv=0;
        else:
            dosend=0; dorecv=1;
    else:
        CHKERR( MPI_Comm_size(comm, &size) )
        CHKERR( MPI_Comm_rank(comm, &rank) )
        if root == rank:
            dosend=1; dorecv=1;
        else:
            dosend=0; dorecv=1;
    #
    cdef object tmps = None
    cdef object rmsg = None
    cdef object tmp1
    #
    if dosend: tmp1 = allocate_count_displ(size, &scounts, &sdispls)
    if dosend: tmps = pickle.dumpv(sendobj, &sbuf, size, scounts, sdispls)
    with PyMPI_Lock(comm, "scatter"):
        with nogil: CHKERR( MPI_Scatter(
            scounts, 1, MPI_INT,
            &rcount, 1, MPI_INT,
            root, comm) )
        if dorecv: rmsg = pickle.alloc(&rbuf, rcount)
        with nogil: CHKERR( MPI_Scatterv(
            sbuf, scounts, sdispls, stype,
            rbuf, rcount,           rtype,
            root, comm) )
    if dorecv: rmsg = pickle.load(rbuf, rcount)
    #
    return rmsg


cdef object PyMPI_allgather(object sendobj, MPI_Comm comm):
    cdef Pickle pickle = PyMPI_PICKLE
    #
    cdef void *sbuf = NULL
    cdef int scount = 0
    cdef MPI_Datatype stype = MPI_BYTE
    cdef void *rbuf = NULL
    cdef int *rcounts = NULL
    cdef int *rdispls = NULL
    cdef MPI_Datatype rtype = MPI_BYTE
    #
    cdef int inter=0, size=0
    CHKERR( MPI_Comm_test_inter(comm, &inter) )
    if inter:
        CHKERR( MPI_Comm_remote_size(comm, &size) )
    else:
        CHKERR( MPI_Comm_size(comm, &size) )
    #
    cdef object tmps = None
    cdef object rmsg = None
    cdef object tmp1
    #
    tmp1 = allocate_count_displ(size, &rcounts, &rdispls)
    tmps = pickle.dump(sendobj, &sbuf, &scount)
    with PyMPI_Lock(comm, "allgather"):
        with nogil: CHKERR( MPI_Allgather(
            &scount, 1, MPI_INT,
            rcounts, 1, MPI_INT,
            comm) )
        rmsg = pickle.allocv(&rbuf, size, rcounts, rdispls)
        with nogil: CHKERR( MPI_Allgatherv(
            sbuf, scount,           stype,
            rbuf, rcounts, rdispls, rtype,
            comm) )
    rmsg = pickle.loadv(rbuf, size, rcounts, rdispls)
    #
    return rmsg


cdef object PyMPI_alltoall(object sendobj, MPI_Comm comm):
    cdef Pickle pickle = PyMPI_PICKLE
    #
    cdef void *sbuf = NULL
    cdef int *scounts = NULL
    cdef int *sdispls = NULL
    cdef MPI_Datatype stype = MPI_BYTE
    cdef void *rbuf = NULL
    cdef int *rcounts = NULL
    cdef int *rdispls = NULL
    cdef MPI_Datatype rtype = MPI_BYTE
    #
    cdef int inter=0, size=0
    CHKERR( MPI_Comm_test_inter(comm, &inter) )
    if inter:
        CHKERR( MPI_Comm_remote_size(comm, &size) )
    else:
        CHKERR( MPI_Comm_size(comm, &size) )
    #
    cdef object tmps = None
    cdef object rmsg = None
    cdef object tmp1, tmp2
    #
    tmp1 = allocate_count_displ(size, &scounts, &sdispls)
    tmp2 = allocate_count_displ(size, &rcounts, &rdispls)
    tmps = pickle.dumpv(sendobj, &sbuf, size, scounts, sdispls)
    with PyMPI_Lock(comm, "alltoall"):
        with nogil: CHKERR( MPI_Alltoall(
            scounts, 1, MPI_INT,
            rcounts, 1, MPI_INT,
            comm) )
        rmsg = pickle.allocv(&rbuf, size, rcounts, rdispls)
        with nogil: CHKERR( MPI_Alltoallv(
            sbuf, scounts, sdispls, stype,
            rbuf, rcounts, rdispls, rtype,
            comm) )
    rmsg = pickle.loadv(rbuf, size, rcounts, rdispls)
    #
    return rmsg


cdef object PyMPI_neighbor_allgather(object sendobj, MPI_Comm comm):
    cdef Pickle pickle = PyMPI_PICKLE
    #
    cdef void *sbuf = NULL
    cdef int scount = 0
    cdef MPI_Datatype stype = MPI_BYTE
    cdef void *rbuf = NULL
    cdef int *rcounts = NULL
    cdef int *rdispls = NULL
    cdef MPI_Datatype rtype = MPI_BYTE
    #
    cdef int i=0, rsize=0
    comm_neighbors_count(comm, &rsize, NULL)
    #
    cdef object tmps = None
    cdef object rmsg = None
    cdef object tmp1
    #
    tmp1 = allocate_count_displ(rsize, &rcounts, &rdispls)
    for i from 0 <= i < rsize: rcounts[i] = 0
    tmps = pickle.dump(sendobj, &sbuf, &scount)
    with PyMPI_Lock(comm, "neighbor_allgather"):
        with nogil: CHKERR( MPI_Neighbor_allgather(
            &scount, 1, MPI_INT,
            rcounts, 1, MPI_INT,
            comm) )
        rmsg = pickle.allocv(&rbuf, rsize, rcounts, rdispls)
        with nogil: CHKERR( MPI_Neighbor_allgatherv(
            sbuf, scount,           stype,
            rbuf, rcounts, rdispls, rtype,
            comm) )
    rmsg = pickle.loadv(rbuf, rsize, rcounts, rdispls)
    #
    return rmsg


cdef object PyMPI_neighbor_alltoall(object sendobj, MPI_Comm comm):
    cdef Pickle pickle = PyMPI_PICKLE
    #
    cdef void *sbuf = NULL
    cdef int *scounts = NULL
    cdef int *sdispls = NULL
    cdef MPI_Datatype stype = MPI_BYTE
    cdef void *rbuf = NULL
    cdef int *rcounts = NULL
    cdef int *rdispls = NULL
    cdef MPI_Datatype rtype = MPI_BYTE
    #
    cdef int i=0, ssize=0, rsize=0
    comm_neighbors_count(comm, &rsize, &ssize)
    #
    cdef object tmps = None
    cdef object rmsg = None
    cdef object tmp1, tmp2
    #
    tmp1 = allocate_count_displ(ssize, &scounts, &sdispls)
    tmp2 = allocate_count_displ(rsize, &rcounts, &rdispls)
    for i from 0 <= i < rsize: rcounts[i] = 0
    tmps = pickle.dumpv(sendobj, &sbuf, ssize, scounts, sdispls)
    with PyMPI_Lock(comm, "neighbor_alltoall"):
        with nogil: CHKERR( MPI_Neighbor_alltoall(
            scounts, 1, MPI_INT,
            rcounts, 1, MPI_INT,
            comm) )
        rmsg = pickle.allocv(&rbuf, rsize, rcounts, rdispls)
        with nogil: CHKERR( MPI_Neighbor_alltoallv(
            sbuf, scounts, sdispls, stype,
            rbuf, rcounts, rdispls, rtype,
            comm) )
    rmsg = pickle.loadv(rbuf, rsize, rcounts, rdispls)
    #
    return rmsg

# -----------------------------------------------------------------------------

cdef inline object _py_reduce(object seq, object op):
    if seq is None: return None
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t n = len(seq)
    cdef object res = seq[0]
    for i from 1 <= i < n:
        res = op(res, seq[i])
    return res

cdef inline object _py_scan(object seq, object op):
    if seq is None: return None
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t n = len(seq)
    for i from 1 <= i < n:
        seq[i] = op(seq[i-1], seq[i])
    return seq

cdef inline object _py_exscan(object seq, object op):
    if seq is None: return None
    seq = _py_scan(seq, op)
    seq.pop(-1)
    seq.insert(0, None)
    return seq

cdef object PyMPI_reduce_naive(object sendobj, object op,
                               int root, MPI_Comm comm):
    cdef object items = PyMPI_gather(sendobj, root, comm)
    return _py_reduce(items, op)

cdef object PyMPI_allreduce_naive(object sendobj, object op, MPI_Comm comm):
    cdef object items = PyMPI_allgather(sendobj, comm)
    return _py_reduce(items, op)

cdef object PyMPI_scan_naive(object sendobj, object op, MPI_Comm comm):
    cdef object items = PyMPI_gather(sendobj, 0, comm)
    items = _py_scan(items, op)
    return PyMPI_scatter(items, 0, comm)

cdef object PyMPI_exscan_naive(object sendobj, object op, MPI_Comm comm):
    cdef object items = PyMPI_gather(sendobj, 0, comm)
    items = _py_exscan(items, op)
    return PyMPI_scatter(items, 0, comm)

# -----

cdef inline object PyMPI_copy(object obj):
    cdef Pickle pickle = PyMPI_PICKLE
    cdef void *buf = NULL
    cdef int count = 0
    obj = pickle.dump(obj, &buf, &count)
    return pickle.load(buf, count)

cdef object PyMPI_send_p2p(object obj, int dst, int tag, MPI_Comm comm):
    cdef Pickle pickle = PyMPI_PICKLE
    cdef void *sbuf = NULL
    cdef int scount = 0
    cdef MPI_Datatype stype = MPI_BYTE
    obj = pickle.dump(obj, &sbuf, &scount)
    with nogil: CHKERR( MPI_Send(&scount, 1, MPI_INT, dst, tag, comm) )
    with nogil: CHKERR( MPI_Send(sbuf, scount, stype, dst, tag, comm) )
    return None

cdef object PyMPI_recv_p2p(int src, int tag, MPI_Comm comm):
    cdef Pickle pickle = PyMPI_PICKLE
    cdef void *rbuf = NULL
    cdef int rcount = 0
    cdef MPI_Datatype rtype = MPI_BYTE
    cdef MPI_Status *status = MPI_STATUS_IGNORE
    cdef object obj
    with nogil: CHKERR( MPI_Recv(&rcount, 1, MPI_INT, src, tag, comm, status) )
    obj = pickle.alloc(&rbuf, rcount)
    with nogil: CHKERR( MPI_Recv(rbuf, rcount, rtype, src, tag, comm, status) )
    return pickle.load(rbuf, rcount)

cdef object PyMPI_sendrecv_p2p(object obj,
                               int dst, int stag,
                               int src, int rtag,
                               MPI_Comm comm):
    cdef Pickle pickle = PyMPI_PICKLE
    cdef void *sbuf = NULL, *rbuf = NULL
    cdef int scount = 0, rcount = 0
    cdef MPI_Datatype dtype = MPI_BYTE
    cdef object tmps = pickle.dump(obj, &sbuf, &scount)
    with nogil: CHKERR( MPI_Sendrecv(&scount, 1, MPI_INT, dst, stag,
                                     &rcount, 1, MPI_INT, src, rtag,
                                     comm, MPI_STATUS_IGNORE) )
    cdef object robj = pickle.alloc(&rbuf, rcount)
    with nogil: CHKERR( MPI_Sendrecv(sbuf, scount, dtype, dst, stag,
                                     rbuf, rcount, dtype, src, rtag,
                                     comm, MPI_STATUS_IGNORE) )
    return pickle.load(rbuf, rcount)

cdef object PyMPI_bcast_p2p(object obj, int root, MPI_Comm comm):
    cdef Pickle pickle = PyMPI_PICKLE
    cdef void *buf = NULL
    cdef int count = 0
    cdef MPI_Datatype dtype = MPI_BYTE
    cdef int rank = MPI_PROC_NULL
    CHKERR( MPI_Comm_rank(comm, &rank) )
    if root == rank: obj = pickle.dump(obj, &buf, &count)
    with PyMPI_Lock(comm, "@bcast_p2p@"):
        with nogil: CHKERR( MPI_Bcast(&count, 1, MPI_INT, root, comm) )
        if root != rank: obj = pickle.alloc(&buf, count)
        with nogil: CHKERR( MPI_Bcast(buf, count, dtype, root, comm) )
    return pickle.load(buf, count)

cdef object PyMPI_reduce_p2p(object sendobj, object op, int root,
                             MPI_Comm comm, int tag):
    # Get communicator size and rank
    cdef int size = MPI_UNDEFINED
    cdef int rank = MPI_PROC_NULL
    CHKERR( MPI_Comm_size(comm, &size) )
    CHKERR( MPI_Comm_rank(comm, &rank) )
    # Check root argument
    if root < 0 or root >= size:
        <void>MPI_Comm_call_errhandler(comm, MPI_ERR_ROOT)
        raise MPIException(MPI_ERR_ROOT)
    if size == 1: sendobj = PyMPI_copy(sendobj)
    #
    cdef object result = sendobj
    cdef object tmp
    # Compute reduction at process 0
    cdef unsigned int umask = <unsigned int> 1
    cdef unsigned int usize = <unsigned int> size
    cdef unsigned int urank = <unsigned int> rank
    cdef int target = 0
    while umask < usize:
        if (umask & urank) != 0:
            target = <int> ((urank & ~umask) % usize)
            PyMPI_send_p2p(result, target, tag, comm)
        else:
            target = <int> (urank | umask)
            if target < size:
                tmp = PyMPI_recv_p2p(target, tag, comm)
                result = op(result, tmp)
        umask <<= 1
    # Send reduction to root
    if root != 0:
        if rank == 0:
            result = PyMPI_send_p2p(result, root, tag, comm)
        elif rank == root:
            result = PyMPI_recv_p2p(0, tag, comm)
    if rank != root:
        result = None
    #
    return result

cdef object PyMPI_scan_p2p(object sendobj, object op,
                           MPI_Comm comm, int tag):
    # Get communicator size and rank
    cdef int size = MPI_UNDEFINED
    cdef int rank = MPI_PROC_NULL
    CHKERR( MPI_Comm_size(comm, &size) )
    CHKERR( MPI_Comm_rank(comm, &rank) )
    if size == 1: sendobj = PyMPI_copy(sendobj)
    #
    cdef object result  = sendobj
    cdef object partial = result
    cdef object tmp
    # Compute prefix op
    cdef unsigned int umask = <unsigned int> 1
    cdef unsigned int usize = <unsigned int> size
    cdef unsigned int urank = <unsigned int> rank
    cdef int target = 0
    while umask < usize:
        target = <int> (urank ^ umask)
        if target < size:
            tmp = PyMPI_sendrecv_p2p(partial, target, tag,
                                     target, tag, comm)
            if rank > target:
                partial = op(tmp, partial)
                result = op(tmp, result)
            else:
                tmp = op(partial, tmp)
                partial = tmp
        umask <<= 1
    #
    return result

cdef object PyMPI_exscan_p2p(object sendobj, object op,
                             MPI_Comm comm, int tag):
    # Get communicator size and rank
    cdef int size = MPI_UNDEFINED
    cdef int rank = MPI_PROC_NULL
    CHKERR( MPI_Comm_size(comm, &size) )
    CHKERR( MPI_Comm_rank(comm, &rank) )
    #
    cdef object result  = sendobj
    cdef object partial = result
    cdef object tmp
    # Compute prefix reduction
    cdef unsigned int umask = <unsigned int> 1
    cdef unsigned int usize = <unsigned int> size
    cdef unsigned int urank = <unsigned int> rank
    cdef unsigned int uflag = <unsigned int> 0
    cdef int target = 0
    while umask < usize:
        target = <int> (urank ^ umask)
        if target < size:
            tmp = PyMPI_sendrecv_p2p(partial, target, tag,
                                     target, tag, comm)
            if rank > target:
                partial = op(tmp, partial)
                if rank != 0:
                    if uflag == 0:
                        result = tmp; uflag = 1
                    else:
                        result = op(tmp, result)
            else:
                tmp = op(partial, tmp)
                partial = tmp
        umask <<= 1
    #
    if rank == 0:
        result = None
    return result

# -----

cdef extern from *:
    int PyMPI_Commctx_intra(MPI_Comm,MPI_Comm*,int*) nogil
    int PyMPI_Commctx_inter(MPI_Comm,MPI_Comm*,int*,MPI_Comm*,int*) nogil

cdef int PyMPI_Commctx_INTRA(MPI_Comm comm, MPI_Comm *dupcomm, int *tag):
    with PyMPI_Lock(comm, "@commctx_intra"):
        return PyMPI_Commctx_intra(comm, dupcomm, tag)
    return MPI_SUCCESS

cdef int PyMPI_Commctx_INTER(MPI_Comm comm, MPI_Comm *dupcomm, int *tag,
                             MPI_Comm *localcomm, int *low_group):
    with PyMPI_Lock(comm, "@commctx_inter"):
        return PyMPI_Commctx_inter(comm, dupcomm, tag, localcomm, low_group)
    return MPI_SUCCESS


cdef object PyMPI_reduce_intra(object sendobj, object op,
                               int root, MPI_Comm comm):
    cdef int tag = MPI_UNDEFINED
    CHKERR( PyMPI_Commctx_INTRA(comm, &comm, &tag) )
    return PyMPI_reduce_p2p(sendobj, op, root, comm, tag)

cdef object PyMPI_reduce_inter(object sendobj, object op,
                               int root, MPI_Comm comm):
    cdef int tag = MPI_UNDEFINED
    cdef MPI_Comm localcomm = MPI_COMM_NULL
    CHKERR( PyMPI_Commctx_INTER(comm, &comm, &tag, &localcomm, NULL) )
    # Get communicator remote size and rank
    cdef int size = MPI_UNDEFINED
    cdef int rank = MPI_PROC_NULL
    CHKERR( MPI_Comm_remote_size(comm, &size) )
    CHKERR( MPI_Comm_rank(comm, &rank) )
    if root >= 0 and root < size:
        # Reduce in local group and send to remote root
        sendobj = PyMPI_reduce_p2p(sendobj, op, 0, localcomm, tag)
        if rank == 0: PyMPI_send_p2p(sendobj, root, tag, comm)
        return None
    elif root == MPI_ROOT: # Receive from remote group
        return PyMPI_recv_p2p(0, tag, comm)
    elif root == MPI_PROC_NULL: # This process does nothing
        return None
    else: # Wrong root argument
        <void>MPI_Comm_call_errhandler(comm, MPI_ERR_ROOT)
        raise MPIException(MPI_ERR_ROOT)


cdef object PyMPI_allreduce_intra(object sendobj, object op, MPI_Comm comm):
    cdef int tag = MPI_UNDEFINED
    CHKERR( PyMPI_Commctx_INTRA(comm, &comm, &tag) )
    sendobj = PyMPI_reduce_p2p(sendobj, op, 0, comm, tag)
    return PyMPI_bcast_p2p(sendobj, 0, comm)

cdef object PyMPI_allreduce_inter(object sendobj, object op, MPI_Comm comm):
    cdef int tag = MPI_UNDEFINED
    cdef int rank = MPI_PROC_NULL
    cdef MPI_Comm localcomm = MPI_COMM_NULL
    CHKERR( PyMPI_Commctx_INTER(comm, &comm, &tag, &localcomm, NULL) )
    CHKERR( MPI_Comm_rank(comm, &rank) )
    # Reduce in local group, exchange, and broadcast in local group
    sendobj = PyMPI_reduce_p2p(sendobj, op, 0, localcomm, tag)
    if rank == 0:
        sendobj = PyMPI_sendrecv_p2p(sendobj, 0, tag, 0, tag, comm)
    return PyMPI_bcast_p2p(sendobj, 0, localcomm)


cdef object PyMPI_scan_intra(object sendobj, object op, MPI_Comm comm):
    cdef int tag = MPI_UNDEFINED
    CHKERR( PyMPI_Commctx_INTRA(comm, &comm, &tag) )
    return PyMPI_scan_p2p(sendobj, op, comm, tag)

cdef object PyMPI_exscan_intra(object sendobj, object op, MPI_Comm comm):
    cdef int tag = MPI_UNDEFINED
    CHKERR( PyMPI_Commctx_INTRA(comm, &comm, &tag) )
    return PyMPI_exscan_p2p(sendobj, op, comm, tag)

# -----

cdef inline bint comm_is_intra(MPI_Comm comm) nogil except -1:
    cdef int inter = 0
    CHKERR( MPI_Comm_test_inter(comm, &inter) )
    if inter: return 0
    else:     return 1


cdef object PyMPI_reduce(object sendobj, object op, int root, MPI_Comm comm):
    if not options.fast_reduce:
        return PyMPI_reduce_naive(sendobj, op, root, comm)
    elif comm_is_intra(comm):
        return PyMPI_reduce_intra(sendobj, op, root, comm)
    else:
        return PyMPI_reduce_inter(sendobj, op, root, comm)


cdef object PyMPI_allreduce(object sendobj, object op, MPI_Comm comm):
    if not options.fast_reduce:
        return PyMPI_allreduce_naive(sendobj, op, comm)
    elif comm_is_intra(comm):
        return PyMPI_allreduce_intra(sendobj, op, comm)
    else:
        return PyMPI_allreduce_inter(sendobj, op, comm)


cdef object PyMPI_scan(object sendobj, object op, MPI_Comm comm):
    if not options.fast_reduce:
        return PyMPI_scan_naive(sendobj, op, comm)
    else:
        return PyMPI_scan_intra(sendobj, op, comm)


cdef object PyMPI_exscan(object sendobj, object op, MPI_Comm comm):
    if not options.fast_reduce:
        return PyMPI_exscan_naive(sendobj, op, comm)
    else:
        return PyMPI_exscan_intra(sendobj, op, comm)

# -----------------------------------------------------------------------------
