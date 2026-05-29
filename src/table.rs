use crate::Error;

/// Successor table used by the Manifold k-NN dynamic program.
///
/// Stores later birth indices `j > i` whose insertion pruned the Voronoi cell of point `i`.
/// This implementation uses a Compressed Sparse Row (CSR) style flat array layout
/// to eliminate memory allocations and improve cache locality during query traversals.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SuccessorTable {
    offsets: Vec<usize>,
    successors: Vec<usize>,
}

impl SuccessorTable {
    /// Creates an empty successor table with `len` lists.
    #[must_use]
    #[inline]
    pub fn empty(len: usize) -> Self {
        Self {
            offsets: vec![0; len + 1],
            successors: Vec::new(),
        }
    }

    /// Creates a successor table from already sorted, duplicate-free lists.
    #[inline]
    pub fn try_from_lists(lists: Vec<Vec<usize>>) -> Result<Self, Error> {
        let mut offsets = Vec::with_capacity(lists.len() + 1);
        offsets.push(0);
        let mut total_successors = 0;
        for list in &lists {
            total_successors += list.len();
            offsets.push(total_successors);
        }
        let mut successors = vec![0; total_successors];
        let mut idx = 0;
        for list in lists {
            let len = list.len();
            successors[idx..idx + len].copy_from_slice(&list);
            idx += len;
        }
        let table = Self { offsets, successors };
        table.validate()?;
        Ok(table)
    }

    /// Creates a successor table from lists that may be unsorted or contain
    /// duplicate entries.
    ///
    /// Entries are sorted and deduplicated before validation.
    #[inline]
    pub fn from_lists_normalized(mut lists: Vec<Vec<usize>>) -> Result<Self, Error> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            lists.par_iter_mut().for_each(|list| {
                list.sort_unstable();
                list.dedup();
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for list in &mut lists {
                list.sort_unstable();
                list.dedup();
            }
        }
        Self::try_from_lists(lists)
    }

    /// Builds a complete quadratic successor table containing every edge `i -> j`
    /// for `i < j`.
    ///
    /// This table makes the query algorithm exact for arbitrary point sets, but
    /// it costs `O(n^2)` memory and query work in the worst case.
    #[must_use]
    pub fn complete(len: usize) -> Self {
        let mut offsets = Vec::with_capacity(len + 1);
        offsets.push(0);
        let mut total = 0;
        for owner in 0..len {
            total += len.saturating_sub(owner + 1);
            offsets.push(total);
        }

        let mut successors = vec![0; total];
        let mut idx = 0;
        for owner in 0..len {
            for successor in (owner + 1)..len {
                successors[idx] = successor;
                idx += 1;
            }
        }
        Self { offsets, successors }
    }

    /// Builds a successor table from insertion-time neighbor lists.
    ///
    /// `neighbors_at_insertion[j]` contains earlier indices `i < j` adjacent to
    /// `j` at insertion time. This constructor appends `j` to each list `i`.
    pub fn from_insertion_neighbors(
        len: usize,
        mut neighbors_at_insertion: Vec<Vec<usize>>,
    ) -> Result<Self, Error> {
        if neighbors_at_insertion.len() != len {
            return Err(Error::TableLengthMismatch {
                points: len,
                lists: neighbors_at_insertion.len(),
            });
        }

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            neighbors_at_insertion.par_iter_mut().for_each(|neighbors| {
                neighbors.sort_unstable();
                neighbors.dedup();
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for neighbors in &mut neighbors_at_insertion {
                neighbors.sort_unstable();
                neighbors.dedup();
            }
        }

        // Count successor list sizes
        let mut counts = vec![0; len];
        for (inserted, neighbors) in neighbors_at_insertion.iter().enumerate() {
            for &neighbor in neighbors {
                if neighbor >= inserted {
                    return Err(Error::InvalidInsertionNeighbor { inserted, neighbor });
                }
                counts[neighbor] += 1;
            }
        }

        // Compute offsets
        let mut offsets = Vec::with_capacity(len + 1);
        offsets.push(0);
        let mut total = 0;
        for &count in &counts {
            total += count;
            offsets.push(total);
        }

        // Populate successors using write cursors to keep track of write positions.
        let mut write_cursors = offsets[..len].to_vec();
        let mut successors = vec![0; total];

        for (inserted, neighbors) in neighbors_at_insertion.into_iter().enumerate() {
            for neighbor in neighbors {
                let pos = &mut write_cursors[neighbor];
                successors[*pos] = inserted;
                *pos += 1;
            }
        }

        let table = Self { offsets, successors };
        table.validate()?;
        Ok(table)
    }

    /// Returns the number of successor lists.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Returns `true` when there are no successor lists.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns all successor lists as an iterator over slices.
    #[must_use]
    #[inline]
    pub fn lists(&self) -> impl Iterator<Item = &[usize]> + '_ {
        (0..self.len()).map(move |owner| self.list(owner))
    }

    /// Returns a successor list by owner index.
    ///
    /// # Panics
    ///
    /// Panics if `owner >= self.len()`.
    #[must_use]
    #[inline]
    pub fn list(&self, owner: usize) -> &[usize] {
        let start = self.offsets[owner];
        let end = self.offsets[owner + 1];
        &self.successors[start..end]
    }

    /// Appends a new empty list. This is used when inserting a new point.
    #[inline]
    pub fn push_empty_list(&mut self) {
        self.offsets.push(self.successors.len());
    }

    /// Inserts successor edge `owner -> successor` while preserving sorted order.
    ///
    /// Returns `true` if an edge was inserted and `false` if it was already
    /// present.
    pub fn insert_successor(&mut self, owner: usize, successor: usize) -> Result<bool, Error> {
        self.validate_successor(owner, successor)?;
        let start = self.offsets[owner];
        let end = self.offsets[owner + 1];
        let list = &self.successors[start..end];
        match list.binary_search(&successor) {
            Ok(_) => Ok(false),
            Err(pos) => {
                let insert_pos = start + pos;
                self.successors.insert(insert_pos, successor);
                for offset in &mut self.offsets[owner + 1..] {
                    *offset += 1;
                }
                Ok(true)
            }
        }
    }

    /// Removes all occurrences of `successor` from every list.
    ///
    /// Returns the number of removed references.
    pub fn remove_references_to(&mut self, successor: usize) -> usize {
        let mut removed = 0;
        let mut write_idx = 0;
        let original_offsets = self.offsets.clone();

        for owner in 0..self.len() {
            let start = original_offsets[owner];
            let end = original_offsets[owner + 1];
            for read_idx in start..end {
                let val = self.successors[read_idx];
                if val == successor {
                    removed += 1;
                } else {
                    self.successors[write_idx] = val;
                    write_idx += 1;
                }
            }
            self.offsets[owner + 1] = write_idx;
        }
        self.successors.truncate(write_idx);
        removed
    }

    /// Clears one successor list.
    pub fn clear_list(&mut self, owner: usize) -> Result<(), Error> {
        if owner >= self.len() {
            return Err(Error::InvalidIndex {
                index: owner,
                len: self.len(),
            });
        }
        let start = self.offsets[owner];
        let end = self.offsets[owner + 1];
        let len_to_remove = end - start;
        if len_to_remove > 0 {
            self.successors.drain(start..end);
            for offset in &mut self.offsets[owner + 1..] {
                *offset -= len_to_remove;
            }
        }
        Ok(())
    }

    /// Clears every successor list while preserving table length.
    pub fn clear_all(&mut self) {
        self.successors.clear();
        self.offsets.fill(0);
    }

    /// Validates the table against its own length.
    #[inline]
    pub fn validate(&self) -> Result<(), Error> {
        self.validate_for_len(self.len())
    }

    /// Validates the table for a point array of length `len`.
    pub fn validate_for_len(&self, len: usize) -> Result<(), Error> {
        if self.len() != len {
            return Err(Error::TableLengthMismatch {
                points: len,
                lists: self.len(),
            });
        }

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            (0..len).into_par_iter().try_for_each(|owner| {
                let start = self.offsets[owner];
                let end = self.offsets[owner + 1];
                let list = &self.successors[start..end];
                let mut previous = None;
                for &successor in list {
                    if successor <= owner || successor >= len {
                        return Err(Error::InvalidSuccessor {
                            owner,
                            successor,
                            len,
                        });
                    }

                    if let Some(prev) = previous {
                        if successor == prev {
                            return Err(Error::DuplicateSuccessor { owner, successor });
                        }
                        if successor < prev {
                            return Err(Error::UnsortedSuccessorList {
                                owner,
                                previous: prev,
                                current: successor,
                            });
                        }
                    }
                    previous = Some(successor);
                }
                Ok(())
            })
        }
        #[cfg(not(feature = "parallel"))]
        {
            for owner in 0..len {
                let start = self.offsets[owner];
                let end = self.offsets[owner + 1];
                let list = &self.successors[start..end];
                let mut previous = None;
                for &successor in list {
                    if successor <= owner || successor >= len {
                        return Err(Error::InvalidSuccessor {
                            owner,
                            successor,
                            len,
                        });
                    }

                    if let Some(prev) = previous {
                        if successor == prev {
                            return Err(Error::DuplicateSuccessor { owner, successor });
                        }
                        if successor < prev {
                            return Err(Error::UnsortedSuccessorList {
                                owner,
                                previous: prev,
                                current: successor,
                            });
                        }
                    }
                    previous = Some(successor);
                }
            }
            Ok(())
        }
    }

    fn validate_successor(&self, owner: usize, successor: usize) -> Result<(), Error> {
        if owner >= self.len() {
            return Err(Error::InvalidIndex {
                index: owner,
                len: self.len(),
            });
        }
        if successor <= owner || successor >= self.len() {
            return Err(Error::InvalidSuccessor {
                owner,
                successor,
                len: self.len(),
            });
        }
        Ok(())
    }
}
