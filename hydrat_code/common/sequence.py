import hydrat
import numpy
import scipy.sparse

"""
Convert sequencing information from a compact list-of-lists-of-indices representation
to a sparse matrix representation suitable for computation (and back).
Single-item sequences should not be present
"""
def sequence2matrix(sequence, size=None):
  if size is None:
    # Try to infer size of collection from the largest identifier in sequence
    size = max( max(s) for s in sequence) + 1
  matrix = scipy.sparse.dok_matrix((size, size), dtype='bool')
  for subseq in sequence:
    if len(subseq) == 1: raise ValueError, "Do not support single-item sequences"
    for i in xrange(len(subseq)-1):
      parent, child = subseq[i:i+2]
      matrix[parent, child] = True
    
  return scipy.sparse.csr_matrix(matrix, dtype=matrix.dtype)
  

# TODO: Test this implementation. If satisfied, replace matrix2sequence with this.
def matrix2seq(matrix):
  # Identify all the first instance in sequences
  first_instance = numpy.nonzero(matrix.sum(0)==0)[1].tolist()[0]
  seq = []
  for i in first_instance:
    subseq = [ i ]
    children = matrix[i].nonzero()[1]
    while(len(children) > 0):
      if len(children) > 1:
        raise ValueError, "Mutliple children- Not a sequence!"
      i = children[0]
      subseq.append(i)
      children = matrix[i].nonzero()[1]
    seq.append(subseq)
  return seq

def matrix2sequence(matrix):
  # NOTE: This assumes that the child index is always greater than
  # the parent index
  # TODO: FIX THIS. Use a first-post-follow approach.
  matrix = scipy.sparse.dok_matrix(matrix, dtype=matrix.dtype)
  pairs = sorted(link for link,b in matrix.items())
  seq = []
  i = iter(pairs)
  subseq = list(i.next())
  for parent, child in i:
    if subseq[-1] == parent:
      subseq.append(child)
    else:
      seq.append(scipy.array(subseq, dtype='uint64'))
      subseq = [parent, child]
  seq.append(scipy.array(subseq, dtype='uint64'))
  return seq

def topological_sort(sequence):
  sq = sequence.tolil()
  ordering = []
  new_nonzeros = [-1]
  while len(new_nonzeros) > 0:
    parent_count = sq.sum(axis=0)
    nonzero_indices = numpy.array(parent_count == 0).nonzero()[1]
    new_nonzeros = sorted(set(nonzero_indices) - set(ordering))
    ordering.extend(new_nonzeros)
    sq[new_nonzeros,:] = 0
  if sq.sum() > 0:
    raise ValueError, "Contains Cycles"
  return ordering

if __name__ == "__main__":
  seq = [[1,2,3],[4,5],[6,7,8,9,10,11,12]]
  mat = sequence2follow(seq)
  seq2 = follow2sequence(mat)
  mat2 = sequence2follow(seq2)
  assert mat.todok().items() == mat2.todok().items()
