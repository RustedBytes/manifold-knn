use crate::Error;

/// Successor table used by the Manifold k-NN dynamic program.
///
/// List `i` stores later birth indices `j > i` whose insertion pruned the
/// Voronoi cell of point `i`. In a Delaunay-based construction, when `j` is
/// inserted, append `j` to every earlier Delaunay neighbor of `j`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SuccessorTable {
    lists: Vec<Vec<usize>>,
}

impl SuccessorTable {
    /// Creates an empty successor table with `len` lists.
    #[must_use]
    pub fn empty(len: usize) -> Self {
        Self {
            lists: vec![Vec::new(); len],
        }
    }

    /// Creates a successor table from already sorted, duplicate-free lists.
    pub fn try_from_lists(lists: Vec<Vec<usize>>) -> Result<Self, Error> {
        let table = Self { lists };
        table.validate()?;
        Ok(table)
    }

    /// Creates a successor table from lists that may be unsorted or contain
    /// duplicate entries.
    ///
    /// Entries are sorted and deduplicated before validation.
    pub fn from_lists_normalized(mut lists: Vec<Vec<usize>>) -> Result<Self, Error> {
        for list in &mut lists {
            list.sort_unstable();
            list.dedup();
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
        let mut lists = Vec::with_capacity(len);
        for owner in 0..len {
            let mut list = Vec::with_capacity(len.saturating_sub(owner + 1));
            for successor in (owner + 1)..len {
                list.push(successor);
            }
            lists.push(list);
        }
        Self { lists }
    }

    /// Builds a successor table from insertion-time neighbor lists.
    ///
    /// `neighbors_at_insertion[j]` contains earlier indices `i < j` adjacent to
    /// `j` at insertion time. This constructor appends `j` to each list `i`.
    pub fn from_insertion_neighbors(
        len: usize,
        neighbors_at_insertion: Vec<Vec<usize>>,
    ) -> Result<Self, Error> {
        if neighbors_at_insertion.len() != len {
            return Err(Error::TableLengthMismatch {
                points: len,
                lists: neighbors_at_insertion.len(),
            });
        }

        let mut table = Self::empty(len);
        for (inserted, neighbors) in neighbors_at_insertion.into_iter().enumerate() {
            let mut neighbors = neighbors;
            neighbors.sort_unstable();
            neighbors.dedup();
            for neighbor in neighbors {
                if neighbor >= inserted {
                    return Err(Error::InvalidInsertionNeighbor { inserted, neighbor });
                }
                table.lists[neighbor].push(inserted);
            }
        }
        table.validate()?;
        Ok(table)
    }

    /// Returns the number of successor lists.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lists.len()
    }

    /// Returns `true` when there are no successor lists.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lists.is_empty()
    }

    /// Returns all successor lists.
    #[must_use]
    pub fn lists(&self) -> &[Vec<usize>] {
        &self.lists
    }

    /// Returns a successor list by owner index.
    ///
    /// # Panics
    ///
    /// Panics if `owner >= self.len()`.
    #[must_use]
    pub fn list(&self, owner: usize) -> &[usize] {
        &self.lists[owner]
    }

    /// Returns a mutable successor list by owner index.
    ///
    /// # Panics
    ///
    /// Panics if `owner >= self.len()`.
    #[must_use]
    pub fn list_mut(&mut self, owner: usize) -> &mut Vec<usize> {
        &mut self.lists[owner]
    }

    /// Appends a new empty list. This is used when inserting a new point.
    pub fn push_empty_list(&mut self) {
        self.lists.push(Vec::new());
    }

    /// Inserts successor edge `owner -> successor` while preserving sorted order.
    ///
    /// Returns `true` if an edge was inserted and `false` if it was already
    /// present.
    pub fn insert_successor(&mut self, owner: usize, successor: usize) -> Result<bool, Error> {
        self.validate_successor(owner, successor)?;
        let list = &mut self.lists[owner];
        match list.binary_search(&successor) {
            Ok(_) => Ok(false),
            Err(position) => {
                list.insert(position, successor);
                Ok(true)
            }
        }
    }

    /// Removes all occurrences of `successor` from every list.
    ///
    /// Returns the number of removed references.
    pub fn remove_references_to(&mut self, successor: usize) -> usize {
        let mut removed = 0;
        for list in &mut self.lists {
            let old_len = list.len();
            list.retain(|&value| value != successor);
            removed += old_len - list.len();
        }
        removed
    }

    /// Clears one successor list.
    pub fn clear_list(&mut self, owner: usize) -> Result<(), Error> {
        if owner >= self.lists.len() {
            return Err(Error::InvalidIndex {
                index: owner,
                len: self.lists.len(),
            });
        }
        self.lists[owner].clear();
        Ok(())
    }

    /// Clears every successor list while preserving table length.
    pub fn clear_all(&mut self) {
        for list in &mut self.lists {
            list.clear();
        }
    }

    /// Validates the table against its own length.
    pub fn validate(&self) -> Result<(), Error> {
        self.validate_for_len(self.lists.len())
    }

    /// Validates the table for a point array of length `len`.
    pub fn validate_for_len(&self, len: usize) -> Result<(), Error> {
        if self.lists.len() != len {
            return Err(Error::TableLengthMismatch {
                points: len,
                lists: self.lists.len(),
            });
        }

        for (owner, list) in self.lists.iter().enumerate() {
            let mut previous = None;
            for &successor in list {
                if successor <= owner || successor >= len {
                    return Err(Error::InvalidSuccessor {
                        owner,
                        successor,
                        len,
                    });
                }

                if let Some(previous) = previous {
                    if successor == previous {
                        return Err(Error::DuplicateSuccessor { owner, successor });
                    }
                    if successor < previous {
                        return Err(Error::UnsortedSuccessorList {
                            owner,
                            previous,
                            current: successor,
                        });
                    }
                }
                previous = Some(successor);
            }
        }
        Ok(())
    }

    fn validate_successor(&self, owner: usize, successor: usize) -> Result<(), Error> {
        if owner >= self.lists.len() {
            return Err(Error::InvalidIndex {
                index: owner,
                len: self.lists.len(),
            });
        }
        if successor <= owner || successor >= self.lists.len() {
            return Err(Error::InvalidSuccessor {
                owner,
                successor,
                len: self.lists.len(),
            });
        }
        Ok(())
    }
}
