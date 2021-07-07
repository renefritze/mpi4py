import sys, os, re
topdir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(topdir, 'conf'))
from mpiscanner import Scanner

LARGECNT_HEAD = """\
#ifndef PyMPI_LARGECNT_H
#define PyMPI_LARGECNT_H

#include <stdlib.h>
#ifndef PyMPI_MALLOC
  #define PyMPI_MALLOC malloc
#endif
#ifndef PyMPI_FREE
  #define PyMPI_FREE free
#endif

#define PyMPICastValue(dsttype, dst, srctype, src)               \\
  do {                                                           \\
    (dst) = (dsttype) (src);                                     \\
    if (((srctype) (dst)) != (src)) {                            \\
      ierr = MPI_ERR_ARG;                                        \\
      (void) MPI_Comm_call_errhandler(MPI_COMM_SELF, ierr);      \\
      goto fn_exit;                                              \\
    }                                                            \\
  } while (0)                                                 /**/

#define PyMPICastArray(dsttype, dst, srctype, src, len)          \\
  do {                                                           \\
    (dst) = NULL;                                                \\
    if ((src) != NULL) {                                         \\
      MPI_Count _n = (len), _m = (_n ? _n : 1), _i;              \\
      size_t _n_alloc = (size_t) _m * sizeof(dsttype);           \\
      (dst) = (dsttype *) PyMPI_MALLOC(_n_alloc);                \\
      if ((dst) == NULL)  {                                      \\
        ierr = MPI_ERR_OTHER;                                    \\
        (void) MPI_Comm_call_errhandler(MPI_COMM_SELF, ierr);    \\
        goto fn_exit;                                            \\
      }                                                          \\
      for (_i = 0; _i < _n; _i++)                                \\
        PyMPICastValue(dsttype, (dst)[_i], srctype, (src)[_i]);  \\
    }                                                            \\
  } while (0)                                                 /**/

#define PyMPIFreeArray(dst)                                      \\
  do {                                                           \\
    if ((dst) != NULL) PyMPI_FREE(dst);                          \\
  } while (0)                                                 /**/

#define PyMPICommSize(comm, n)                                   \\
  do {                                                           \\
    int _inter;                                                  \\
    ierr = MPI_Comm_test_inter(comm, &_inter);                   \\
    if (_inter)                                                  \\
      ierr = MPI_Comm_remote_size((comm), &(n));                 \\
    else                                                         \\
      ierr = MPI_Comm_size((comm), &(n));                        \\
    if (ierr != MPI_SUCCESS) goto fn_exit;                       \\
  } while (0)                                                 /**/

#define PyMPICommLocGroupSize(comm, n)                           \\
  do {                                                           \\
    ierr = MPI_Comm_size((comm), &(n));                          \\
    if (ierr != MPI_SUCCESS) goto fn_exit;                       \\
  } while (0)                                                 /**/

#define PyMPICommNeighborCount(comm, ns, nr)                     \\
  do {                                                           \\
    int _topo, _i, _n; (ns) = (nr) = 0;                          \\
    ierr = MPI_Topo_test((comm), &_topo);                        \\
    if (ierr != MPI_SUCCESS) goto fn_exit;                       \\
    if (_topo == MPI_UNDEFINED) {                                \\
      ierr = MPI_Comm_size((comm), &_n);                         \\
      (ns) = (nr) = _n;                                          \\
    } else if (_topo == MPI_CART) {                              \\
      ierr = MPI_Cartdim_get((comm), &_n);                       \\
      (ns) = (nr) = 2 * _n;                                      \\
    } else if (_topo == MPI_GRAPH) {                             \\
      ierr = MPI_Comm_rank((comm), &_i);                         \\
      ierr = MPI_Graph_neighbors_count(                          \\
               (comm), _i, &_n);                                 \\
      (ns) = (nr) = _n;                                          \\
    } else if (_topo == MPI_DIST_GRAPH) {                        \\
      ierr = MPI_Dist_graph_neighbors_count(                     \\
               (comm), &(nr), &(ns), &_i);                       \\
    }                                                            \\
    if (ierr != MPI_SUCCESS) goto fn_exit;                       \\
  } while (0)                                                 /**/

"""

LARGECNT_TAIL = """\
#endif /* !PyMPI_LARGECNT_H */
"""

LARGECNT_BEGIN = """\
#ifndef PyMPI_HAVE_%(name)s_c
static int
Py%(name)s_c(%(argsdecl)s)
{
"""

LARGECNT_COLLECTIVE = """\
  PyMPICommSize(a%(commid)d, n);
"""

LARGECNT_LOCGROUP = """\
  PyMPICommLocGroupSize(a%(commid)d, n);
"""

LARGECNT_NEIGHBOR = """\
  PyMPICommNeighborCount(a%(commid)d, ns, nr);
"""

LARGECNT_CALL = """\
  ierr = %(name)s(%(argscall)s);
  if (ierr != MPI_SUCCESS) goto fn_exit;
"""

LARGECNT_END = """\
  return ierr;
}
#undef  %(name)s_c
#define %(name)s_c Py%(name)s_c
#endif
"""

def largecount_functions(self):
    p2ps = r'(i?(b|s|r|p)?send(_init)?(recv(_replace)?)?)'
    p2pr = r'(i?m?p?recv(_init)?)'
    p2pc = r'(buffer_(at|de)tach|get_count)'
    coll = r'(i?(bcast|gather(v)?|scatter(v?)|all(gather(v)?|toall(v|w)?)))'
    red  = r'(i?((all)?reduce(_local|(_scatter(_block)?)?)|(ex)?scan)(_init)?)'
    ngh  = r'(i?neighbor_all(gather(v)?|toall(v|w)?))'
    win  = r'(win_(create|allocate(_shared)?|shared_query))'
    rma  = r'(r?(put|get|(get_)?accumulate))'
    io   = r'file_(((i)?(read|write).*)|get_type_extent)'
    largecount = re.compile(
        r'^mpi_(%s)_c$' % '|'.join([
            p2ps, p2pr, p2pc,
            coll, red, ngh,
            win, rma, io,
        ])
    )
    for node in self:
        name = node.name
        nlow = name.lower()
        if largecount.match(nlow):
            yield name[:-2]

def dump_largecnt_h(self, fileobj):
    if isinstance(fileobj, str):
        with open(fileobj, 'w') as f:
            return dump_largecnt_h(self, f)
        return

    def declare(t, v, init=None):
        t = t.strip()
        if t.endswith('[]'):
            t = t[:-2].strip()
            code = '%s *%s' % (t, v)
        elif t.endswith('*'):
            t = t[:-1].strip()
            code = '%s *%s' % (t, v)
        else:
            code = '%s %s' % (t, v)
        if init is not None:
            code += ' = %s' % init
        return code

    def generate(self, name):
        is_neighbor = 'neighbor' in name.lower()
        node1 = self[name+'_c']
        node2 = self[name]

        cargstype1 = node1.cargstype
        cargstype2 = node2.cargstype
        assert len(cargstype1) == len(cargstype2)
        argstype = list(zip(cargstype1, cargstype2))

        convert_array = False
        for (t1, t2) in argstype:
            if t1 != t2:
                if t1 == 'MPI_Count[]' and t2 == 'int[]':
                    convert_array = True
                    break
                if t1 == 'MPI_Aint[]'  and t2 == 'int[]':
                    convert_array = True
                    break
        commid = None
        if convert_array:
            for i, (t1, t2) in enumerate(argstype, start=1):
                if t1 == 'MPI_Comm':
                    commid = i
                    break

        dtypeidx = 0
        argslist = []
        argsinit = []
        argstemp = []
        argsconv = []
        argscall = []
        argsoutp = []
        argsfree = []

        argsinit += ['int ierr']
        if commid is not None:
            if is_neighbor:
                argsinit += ['int ns, nr']
            else:
                argsinit += ['int n']
        for i, (t1, t2) in enumerate(argstype, start=1):
            argslist += [declare(t1, 'a%d' % i)]
            if t1.startswith('MPI_Datatype'):
                dtypeidx += 1
            if t1 == t2:
                argscall += ['a%d' % i]
            else:
                if t1.endswith('[]'):
                    t1, t2, n = t1[:-2], t2[:-2], 'n'
                    argstemp += [declare(t2, '*b%d' % i, 'NULL')]
                    if is_neighbor: n = ('ns', 'nr')[dtypeidx]
                    subs = (t2, i, t1, i, n)
                    argsconv += ['PyMPICastArray(%s, b%d, %s, a%d, %s)' % subs]
                    argsfree += ['PyMPIFreeArray(b%d)' % i]
                    argscall += ['b%d' % i]
                elif t1.endswith('*'):
                    t1, t2 = t1[:-1], t2[:-1]
                    ptr_init = 'a%d ? &b%d : NULL' % (i, i)
                    argstemp += [declare(t2, 'b%d' % i, 0)]
                    argstemp += [declare(t2+'*', 'p%d'% i, ptr_init)]
                    argscall += ['p%d' % i]
                    argsoutp += ['if (a%d) *a%d = b%d' % (i, i, i)]
                else:
                    subs = (t2, i, t1, i)
                    argstemp += [declare(t2, 'b%d' % i)]
                    argsconv += ['PyMPICastValue(%s, b%d, %s, a%d)' % subs]
                    argscall += ['b%d' % i]

        subs = dict(
            name=name,
            argsdecl=(',\n'+' '*(len(name)+5)).join(argslist),
            argscall=', '.join(argscall),
            commid=commid,
        )
        begin = (LARGECNT_BEGIN) % subs
        if commid is None:
            setup = ''
        elif is_neighbor:
            setup = (LARGECNT_NEIGHBOR) % subs
        elif 'reduce_scatter' in name.lower():
            setup = (LARGECNT_LOCGROUP) % subs
        else:
            setup = (LARGECNT_COLLECTIVE) % subs
        call = (LARGECNT_CALL) % subs
        end = (LARGECNT_END) % subs

        tab = '  '
        yield (begin)
        yield (tab + '%s;\n' % '; '.join(argsinit))
        yield (tab + '%s;\n' % '; '.join(argstemp))
        yield (setup)
        for line in argsconv:
            yield (tab + '%s;\n' % line)
        yield (call)
        for line in argsoutp:
            yield (tab + '%s;\n' % line)
        yield (tab+'fn_exit:\n')
        for line in argsfree:
            yield (tab + '%s;\n' % line)
        yield (end)
        yield ('\n')


    count = 0
    fileobj.write(LARGECNT_HEAD)
    for name in largecount_functions(self):
        for code in generate(self, name):
            fileobj.write(code)
        count += 1
    fileobj.write(LARGECNT_TAIL)
    return count


if __name__ == '__main__':
    log = lambda msg: sys.stderr.write(msg + '\n')
    sources = [os.path.join('src', 'mpi4py', 'libmpi.pxd')]
    scanner = Scanner()
    for filename in sources:
        log('parsing file %s' % filename)
        scanner.parse_file(filename)
    log('processed %d declarations' % len(scanner.nodes))

    largecnt_h = os.path.join('src', 'lib-mpi', 'largecnt.h')
    log('writing file %s' % largecnt_h)
    n = dump_largecnt_h(scanner, largecnt_h)
    log('generated %d fallbacks' % n)
