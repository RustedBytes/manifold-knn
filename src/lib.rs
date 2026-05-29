//! Dynamic-programming k-nearest-neighbor queries for birth-ordered point clouds.
//!
//! This crate implements the query-side data structure described by Wang et al.,
//! *Manifold k-NN: Accelerated k-NN Queries for Manifold Point Clouds*.
//!
//! The paper separates the method into two concerns:
//!
//! 1. Maintain a **successor table**. When a point `j` is inserted, append `j` to
//!    every earlier point whose Voronoi cell is pruned by the insertion. In a
//!    Delaunay implementation this means appending `j` to the Delaunay neighbors
//!    of `j` in the prefix triangulation.
//! 2. Answer k-NN queries by first following the dynamic-programming 1-NN path,
//!    then expanding only the successor lists of the best candidates.
//!
//! This crate implements item 2, plus safe helpers for constructing and updating
//! successor tables. With the optional `delaunay-3d` feature it can also build
//! insertion-time successor tables directly from the `delaunay` crate's 3D
//! incremental triangulation API.
//!
//! # Example
//!
//! ```
//! use manifold_knn::ManifoldKnn;
//!
//! let points = vec![
//!     [0.0, 0.0, 0.0],
//!     [1.0, 0.0, 0.0],
//!     [0.0, 1.0, 0.0],
//!     [0.0, 0.0, 1.0],
//! ];
//!
//! // Exact but quadratic fallback: every later point is a successor of every
//! // earlier point. This is useful for tests and small point sets.
//! let index = ManifoldKnn::<3>::from_complete_successors(points)?;
//! let nn = index.knn(&[0.2, 0.1, 0.0], 2)?;
//!
//! assert_eq!(nn.len(), 2);
//! assert_eq!(nn[0].index, 0);
//! # Ok::<(), manifold_knn::Error>(())
//! ```

#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(unsafe_code)]
#![warn(missing_docs)]

mod bounded;
mod error;
mod table;

#[cfg(feature = "delaunay-3d")]
pub mod delaunay3d;

#[cfg(feature = "delaunay-3d")]
pub use delaunay3d::{Delaunay3dKernel, DelaunayManifoldKnn3, DelaunayTriangulation3};

pub use error::Error;
pub use table::SuccessorTable;

use bounded::{BoundedNeighbors, neighbor_cmp};

/// A nearest-neighbor result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    /// Index of the point in the birth-time-ordered point array.
    pub index: usize,
    /// Squared Euclidean distance from the query to the point.
    pub squared_distance: f64,
}

impl Neighbor {
    /// Returns the Euclidean distance from the query to this neighbor.
    #[must_use]
    pub fn distance(self) -> f64 {
        self.squared_distance.sqrt()
    }
}

/// Summary returned by deletion/update operations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DeleteReport {
    /// Number of references to the deleted point that were removed from lists.
    pub removed_references: usize,
    /// Number of new successor edges inserted.
    pub inserted_successors: usize,
    /// Number of requested successor edges that were already present.
    pub already_present: usize,
}

/// Dynamic-programming k-NN index over a fixed-dimensional Euclidean point set.
///
/// `D` is the ambient dimension. For the paper's common point-cloud case use
/// `ManifoldKnn<3>`.
#[derive(Clone, Debug)]
pub struct ManifoldKnn<const D: usize> {
    points: Vec<[f64; D]>,
    successors: SuccessorTable,
    active: Vec<bool>,
}

impl<const D: usize> ManifoldKnn<D> {
    /// Creates an index from points and a successor table.
    ///
    /// The successor table must have one list per point; every entry in list `i`
    /// must be a strictly later birth index `j > i`. Lists must be sorted and
    /// duplicate-free. Use [`SuccessorTable::from_lists_normalized`] when input
    /// lists may be unsorted.
    pub fn new(points: Vec<[f64; D]>, successors: SuccessorTable) -> Result<Self, Error> {
        validate_points(&points)?;
        successors.validate_for_len(points.len())?;
        let active = vec![true; points.len()];
        Ok(Self {
            points,
            successors,
            active,
        })
    }

    /// Builds an exact but quadratic successor table.
    ///
    /// This stores every pair `i < j` as a successor edge `i -> j`. It is useful
    /// as a correctness oracle, for small inputs, and for tests. It does **not**
    /// provide the acceleration of the manifold/Delaunay successor table.
    pub fn from_complete_successors(points: Vec<[f64; D]>) -> Result<Self, Error> {
        let n = points.len();
        Self::new(points, SuccessorTable::complete(n))
    }

    /// Builds an index from insertion-time neighbor lists.
    ///
    /// `neighbors_at_insertion[j]` must contain earlier birth indices `i < j`
    /// that were adjacent to `j` when `j` was inserted. In the paper, these are
    /// the Delaunay neighbors of `p_j` in the prefix triangulation. For every
    /// such neighbor, this constructor appends `j` to the successor list of `i`.
    pub fn from_insertion_neighbors(
        points: Vec<[f64; D]>,
        neighbors_at_insertion: Vec<Vec<usize>>,
    ) -> Result<Self, Error> {
        let successors =
            SuccessorTable::from_insertion_neighbors(points.len(), neighbors_at_insertion)?;
        Self::new(points, successors)
    }

    /// Returns all points in birth order.
    #[must_use]
    pub fn points(&self) -> &[[f64; D]] {
        &self.points
    }

    /// Returns the successor table.
    #[must_use]
    pub fn successors(&self) -> &SuccessorTable {
        &self.successors
    }

    /// Returns a mutable successor table.
    ///
    /// This is exposed for integration with external Delaunay maintenance code.
    /// Call [`SuccessorTable::validate_for_len`] after substantial edits.
    #[must_use]
    pub fn successors_mut(&mut self) -> &mut SuccessorTable {
        &mut self.successors
    }

    /// Number of stored points, including inactive points left by deletion APIs.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` if the index contains no stored points.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Number of active points.
    #[must_use]
    #[inline]
    pub fn active_len(&self) -> usize {
        self.active.iter().filter(|&&is_active| is_active).count()
    }

    /// Returns whether the point at `index` is active.
    #[must_use]
    #[inline]
    pub fn is_active(&self, index: usize) -> bool {
        self.active.get(index).copied().unwrap_or(false)
    }

    /// Iterator over active point indices in birth order.
    #[inline]
    pub fn active_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.active
            .iter()
            .enumerate()
            .filter_map(|(index, &is_active)| is_active.then_some(index))
    }

    /// Inserts a new point and updates successor lists from externally supplied
    /// insertion-time neighbors.
    ///
    /// `prior_neighbors` must contain active earlier indices that are adjacent to
    /// the new point at insertion time. In a faithful implementation of the
    /// paper, these come from incremental Delaunay construction.
    ///
    /// Returns the birth index assigned to the inserted point.
    #[inline]
    pub fn insert_with_neighbors<I>(
        &mut self,
        point: [f64; D],
        prior_neighbors: I,
    ) -> Result<usize, Error>
    where
        I: IntoIterator<Item = usize>,
    {
        validate_point(self.points.len(), &point)?;

        let new_index = self.points.len();
        let mut neighbors: Vec<usize> = prior_neighbors.into_iter().collect();
        neighbors.sort_unstable();
        neighbors.dedup();

        for &neighbor in &neighbors {
            if neighbor >= new_index {
                return Err(Error::InvalidInsertionNeighbor {
                    inserted: new_index,
                    neighbor,
                });
            }
            if !self.active[neighbor] {
                return Err(Error::InactivePoint { index: neighbor });
            }
        }

        self.points.push(point);
        self.active.push(true);
        self.successors.push_empty_list();

        for neighbor in neighbors {
            self.successors.insert_successor(neighbor, new_index)?;
        }

        Ok(new_index)
    }

    /// Returns the nearest active point in the full index.
    #[inline]
    pub fn nearest(&self, query: &[f64; D]) -> Result<Option<Neighbor>, Error> {
        WORKSPACE.with(|ws| {
            let mut workspace = ws.borrow_mut();
            self.nearest_with_workspace(query, &mut workspace)
        })
    }

    /// Returns the nearest active point in the full index using a workspace to avoid allocations.
    #[inline]
    pub fn nearest_with_workspace(
        &self,
        query: &[f64; D],
        workspace: &mut QueryWorkspace,
    ) -> Result<Option<Neighbor>, Error> {
        let slice = self.knn_prefix_with_workspace(query, 1, self.points.len(), workspace)?;
        Ok(slice.first().copied())
    }

    /// Returns up to `k` nearest active points in the full index.
    ///
    /// Results are sorted by squared distance, with birth index used as a stable
    /// tie-breaker.
    #[inline]
    pub fn knn(&self, query: &[f64; D], k: usize) -> Result<Vec<Neighbor>, Error> {
        self.knn_prefix(query, k, self.points.len())
    }

    /// Returns up to `k` nearest active points in the full index using a workspace.
    #[inline]
    pub fn knn_with_workspace<'a>(
        &self,
        query: &[f64; D],
        k: usize,
        workspace: &'a mut QueryWorkspace,
    ) -> Result<&'a [Neighbor], Error> {
        self.knn_prefix_with_workspace(query, k, self.points.len(), workspace)
    }

    /// Returns up to `k` nearest active points whose birth index is `< prefix_len`.
    ///
    /// This is the paper's zero-overhead prefix-subset query. The same successor
    /// table is reused; candidates outside the prefix are ignored.
    #[inline]
    pub fn knn_prefix(
        &self,
        query: &[f64; D],
        k: usize,
        prefix_len: usize,
    ) -> Result<Vec<Neighbor>, Error> {
        WORKSPACE.with(|ws| {
            let mut workspace = ws.borrow_mut();
            let slice = self.knn_prefix_with_workspace(query, k, prefix_len, &mut workspace)?;
            Ok(slice.to_vec())
        })
    }

    /// Returns up to `k` nearest active points whose birth index is `< prefix_len` using a workspace.
    #[inline]
    pub fn knn_prefix_with_workspace<'a>(
        &self,
        query: &[f64; D],
        k: usize,
        prefix_len: usize,
        workspace: &'a mut QueryWorkspace,
    ) -> Result<&'a [Neighbor], Error> {
        self.validate_query_and_prefix(query, prefix_len)?;
        if k == 0 {
            workspace.candidates.reset(0);
            return Ok(&[]);
        }

        let Some(first_active) = self.first_active_before(prefix_len) else {
            workspace.candidates.reset(0);
            return Ok(&[]);
        };

        self.transition_sites_internal_from_index_with_workspace(
            query,
            k,
            prefix_len,
            first_active,
            workspace,
        )?;

        if workspace.processed.len() < self.points.len() {
            workspace.processed.resize(self.points.len(), false);
        }
        if workspace.discovered.len() < self.points.len() {
            workspace.discovered.resize(self.points.len(), false);
        }

        for candidate in workspace.candidates.as_slice() {
            let idx = candidate.index;
            if !workspace.discovered[idx] {
                workspace.discovered[idx] = true;
                workspace.visited_indices.push(idx);
            }
        }

        while let Some(index) = workspace
            .candidates
            .as_slice()
            .iter()
            .find(|candidate| !workspace.processed[candidate.index])
            .map(|candidate| candidate.index)
        {
            workspace.processed[index] = true;

            for &successor in self.successors.list(index) {
                if successor >= prefix_len {
                    break;
                }
                if workspace.discovered[successor] {
                    continue;
                }
                if !self.active[successor] {
                    continue;
                }
                workspace.discovered[successor] = true;
                workspace.visited_indices.push(successor);

                workspace.candidates.insert(Neighbor {
                    index: successor,
                    squared_distance: squared_distance(&self.points[successor], query),
                });
            }
        }

        for &index in &workspace.visited_indices {
            workspace.processed[index] = false;
            workspace.discovered[index] = false;
        }
        workspace.visited_indices.clear();

        Ok(workspace.candidates.as_slice())
    }

    /// Collects transition sites from the dynamic-programming 1-NN path.
    ///
    /// Transition sites are the successive points that become the nearest
    /// neighbor during the birth-ordered insertion history. The returned list is
    /// sorted by query distance and capped to `capacity` entries.
    #[inline]
    pub fn transition_sites(
        &self,
        query: &[f64; D],
        capacity: usize,
    ) -> Result<Vec<Neighbor>, Error> {
        self.transition_sites_prefix(query, capacity, self.points.len())
    }

    /// Collects transition sites using a workspace.
    #[inline]
    pub fn transition_sites_with_workspace<'a>(
        &self,
        query: &[f64; D],
        capacity: usize,
        workspace: &'a mut QueryWorkspace,
    ) -> Result<&'a [Neighbor], Error> {
        self.transition_sites_prefix_with_workspace(query, capacity, self.points.len(), workspace)
    }

    /// Prefix-restricted variant of [`Self::transition_sites`].
    #[inline]
    pub fn transition_sites_prefix(
        &self,
        query: &[f64; D],
        capacity: usize,
        prefix_len: usize,
    ) -> Result<Vec<Neighbor>, Error> {
        WORKSPACE.with(|ws| {
            let mut workspace = ws.borrow_mut();
            let slice = self.transition_sites_prefix_with_workspace(
                query,
                capacity,
                prefix_len,
                &mut workspace,
            )?;
            Ok(slice.to_vec())
        })
    }

    /// Prefix-restricted variant of [`Self::transition_sites`] using a workspace.
    #[inline]
    pub fn transition_sites_prefix_with_workspace<'a>(
        &self,
        query: &[f64; D],
        capacity: usize,
        prefix_len: usize,
        workspace: &'a mut QueryWorkspace,
    ) -> Result<&'a [Neighbor], Error> {
        self.validate_query_and_prefix(query, prefix_len)?;
        self.transition_sites_internal_with_workspace(query, capacity, prefix_len, workspace)?;
        Ok(workspace.candidates.as_slice())
    }

    /// Brute-force k-NN over the active full index.
    ///
    /// This is intended for validation and benchmarking, not production queries.
    #[inline]
    pub fn brute_force_knn(&self, query: &[f64; D], k: usize) -> Result<Vec<Neighbor>, Error> {
        self.brute_force_knn_prefix(query, k, self.points.len())
    }

    /// Brute-force k-NN over active points with birth index `< prefix_len`.
    #[inline]
    pub fn brute_force_knn_prefix(
        &self,
        query: &[f64; D],
        k: usize,
        prefix_len: usize,
    ) -> Result<Vec<Neighbor>, Error> {
        self.validate_query_and_prefix(query, prefix_len)?;
        if k == 0 {
            return Ok(Vec::new());
        }

        let mut neighbors = Vec::new();
        for index in 0..prefix_len {
            if self.active[index] {
                neighbors.push(Neighbor {
                    index,
                    squared_distance: squared_distance(&self.points[index], query),
                });
            }
        }
        neighbors.sort_by(neighbor_cmp);
        neighbors.truncate(k);
        Ok(neighbors)
    }

    /// Marks `index` as deleted and applies externally computed local successor
    /// edges that replace adjacencies created by the deletion.
    ///
    /// This is the safe hook for the paper's local Delaunay deletion algorithm.
    /// The crate validates and applies the successor-table mutations, while the
    /// caller supplies the geometric result of the local retriangulation.
    ///
    /// Each pair `(i, j)` in `new_successors` means "insert successor edge
    /// `i -> j`" and must satisfy `i < j`; both endpoints must be active and must
    /// not be the deleted point.
    #[inline]
    pub fn delete_with_new_successors<I>(
        &mut self,
        index: usize,
        new_successors: I,
    ) -> Result<DeleteReport, Error>
    where
        I: IntoIterator<Item = (usize, usize)>,
    {
        self.ensure_active_index(index)?;

        let mut edges: Vec<(usize, usize)> = new_successors.into_iter().collect();
        edges.sort_unstable();
        edges.dedup();

        for &(owner, successor) in &edges {
            self.validate_new_successor_edge(index, owner, successor)?;
        }

        let removed_references = self.successors.remove_references_to(index);
        self.successors.clear_list(index)?;
        self.active[index] = false;

        let mut inserted_successors = 0;
        let mut already_present = 0;
        for (owner, successor) in edges {
            if self.successors.insert_successor(owner, successor)? {
                inserted_successors += 1;
            } else {
                already_present += 1;
            }
        }

        Ok(DeleteReport {
            removed_references,
            inserted_successors,
            already_present,
        })
    }

    /// Deletes a point and rebuilds a complete quadratic successor table over the
    /// remaining active points.
    ///
    /// This is an exact fallback for applications that need correctness after a
    /// deletion but do not yet have local Delaunay-update integration. It discards
    /// the accelerated successor table and should only be used for small or test
    /// data.
    #[inline]
    pub fn delete_rebuild_complete(&mut self, index: usize) -> Result<DeleteReport, Error> {
        self.ensure_active_index(index)?;
        let removed_references = self.successors.remove_references_to(index);
        self.successors.clear_all();
        self.active[index] = false;

        let mut inserted_successors = 0;
        for owner in 0..self.points.len() {
            if !self.active[owner] {
                continue;
            }
            for successor in (owner + 1)..self.points.len() {
                if self.active[successor] {
                    self.successors.insert_successor(owner, successor)?;
                    inserted_successors += 1;
                }
            }
        }

        Ok(DeleteReport {
            removed_references,
            inserted_successors,
            already_present: 0,
        })
    }

    #[inline]
    fn transition_sites_internal_with_workspace(
        &self,
        query: &[f64; D],
        capacity: usize,
        prefix_len: usize,
        workspace: &mut QueryWorkspace,
    ) -> Result<(), Error> {
        if capacity == 0 {
            workspace.candidates.reset(0);
            return Ok(());
        }
        let Some(first_active) = self.first_active_before(prefix_len) else {
            workspace.candidates.reset(capacity);
            return Ok(());
        };
        self.transition_sites_internal_from_index_with_workspace(
            query,
            capacity,
            prefix_len,
            first_active,
            workspace,
        )
    }

    #[inline]
    fn transition_sites_internal_from_index_with_workspace(
        &self,
        query: &[f64; D],
        capacity: usize,
        prefix_len: usize,
        mut current: usize,
        workspace: &mut QueryWorkspace,
    ) -> Result<(), Error> {
        workspace.candidates.reset(capacity);
        if capacity == 0 {
            return Ok(());
        }

        workspace.candidates.insert(Neighbor {
            index: current,
            squared_distance: squared_distance(&self.points[current], query),
        });

        loop {
            let current_distance = squared_distance(&self.points[current], query);
            let mut next = None;

            for &successor in self.successors.list(current) {
                if successor >= prefix_len {
                    break;
                }
                if !self.active[successor] {
                    continue;
                }

                let successor_distance = squared_distance(&self.points[successor], query);
                if is_strictly_better(successor_distance, successor, current_distance, current) {
                    next = Some((successor, successor_distance));
                    break;
                }
            }

            let Some((successor, successor_distance)) = next else {
                break;
            };

            current = successor;
            workspace.candidates.insert(Neighbor {
                index: current,
                squared_distance: successor_distance,
            });
        }

        Ok(())
    }

    #[inline]
    fn validate_query_and_prefix(&self, query: &[f64; D], prefix_len: usize) -> Result<(), Error> {
        validate_query(query)?;
        if prefix_len > self.points.len() {
            return Err(Error::InvalidPrefix {
                prefix_len,
                len: self.points.len(),
            });
        }
        Ok(())
    }

    #[inline]
    fn first_active_before(&self, prefix_len: usize) -> Option<usize> {
        self.active[..prefix_len]
            .iter()
            .position(|&is_active| is_active)
    }

    #[inline]
    fn ensure_active_index(&self, index: usize) -> Result<(), Error> {
        if index >= self.points.len() {
            return Err(Error::InvalidIndex {
                index,
                len: self.points.len(),
            });
        }
        if !self.active[index] {
            return Err(Error::InactivePoint { index });
        }
        Ok(())
    }

    #[inline]
    fn validate_new_successor_edge(
        &self,
        deleted: usize,
        owner: usize,
        successor: usize,
    ) -> Result<(), Error> {
        if owner >= self.points.len() {
            return Err(Error::InvalidIndex {
                index: owner,
                len: self.points.len(),
            });
        }
        if successor >= self.points.len() {
            return Err(Error::InvalidIndex {
                index: successor,
                len: self.points.len(),
            });
        }
        if owner == deleted || successor == deleted || owner >= successor {
            return Err(Error::InvalidSuccessor {
                owner,
                successor,
                len: self.points.len(),
            });
        }
        if !self.active[owner] {
            return Err(Error::InactivePoint { index: owner });
        }
        if !self.active[successor] {
            return Err(Error::InactivePoint { index: successor });
        }
        Ok(())
    }
}

/// Reusable query workspace to avoid allocations during nearest-neighbor queries.
#[derive(Clone, Debug, Default)]
pub struct QueryWorkspace {
    processed: Vec<bool>,
    discovered: Vec<bool>,
    visited_indices: Vec<usize>,
    candidates: BoundedNeighbors,
}

impl QueryWorkspace {
    /// Creates a new empty query workspace.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            processed: Vec::new(),
            discovered: Vec::new(),
            visited_indices: Vec::new(),
            candidates: BoundedNeighbors::new_empty(),
        }
    }
}

thread_local! {
    static WORKSPACE: std::cell::RefCell<QueryWorkspace> = const { std::cell::RefCell::new(QueryWorkspace::new()) };
}

#[inline]
pub(crate) fn validate_points<const D: usize>(points: &[[f64; D]]) -> Result<(), Error> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        points
            .par_iter()
            .enumerate()
            .try_for_each(|(index, point)| validate_point(index, point))
    }
    #[cfg(not(feature = "parallel"))]
    {
        for (index, point) in points.iter().enumerate() {
            validate_point(index, point)?;
        }
        Ok(())
    }
}

#[inline]
pub(crate) fn validate_point<const D: usize>(index: usize, point: &[f64; D]) -> Result<(), Error> {
    for (coordinate, &value) in point.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::NonFiniteCoordinate {
                point: index,
                coordinate,
                value,
            });
        }
    }
    Ok(())
}

#[inline]
pub(crate) fn validate_query<const D: usize>(query: &[f64; D]) -> Result<(), Error> {
    for (coordinate, &value) in query.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::NonFiniteQuery { coordinate, value });
        }
    }
    Ok(())
}

#[cfg(feature = "simd")]
#[inline]
pub(crate) fn squared_distance<const D: usize>(point: &[f64; D], query: &[f64; D]) -> f64 {
    use std::simd::{f64x4, num::SimdFloat};

    if D == 0 {
        return 0.0;
    }

    // Fast path for the most common small dimensions (D=2, D=3, D=4)
    // This covers the vast majority of point-cloud use cases.
    if D <= 4 {
        let mut p = [0.0_f64; 4];
        let mut q = [0.0_f64; 4];
        p[..D].copy_from_slice(&point[..D]);
        q[..D].copy_from_slice(&query[..D]);
        let pv = f64x4::from_array(p);
        let qv = f64x4::from_array(q);
        let diff = pv - qv;
        return (diff * diff).reduce_sum();
    }

    // General case: process 4 elements at a time using SIMD
    let mut sum = 0.0_f64;
    let mut i = 0;

    while i + 4 <= D {
        let pv = f64x4::from_array([point[i], point[i + 1], point[i + 2], point[i + 3]]);
        let qv = f64x4::from_array([query[i], query[i + 1], query[i + 2], query[i + 3]]);
        let diff = pv - qv;
        sum += (diff * diff).reduce_sum();
        i += 4;
    }

    // Handle remaining elements (0-3)
    for j in i..D {
        let delta = point[j] - query[j];
        sum += delta * delta;
    }

    sum
}

#[cfg(not(feature = "simd"))]
#[inline]
pub(crate) fn squared_distance<const D: usize>(point: &[f64; D], query: &[f64; D]) -> f64 {
    let mut sum = 0.0;
    for i in 0..D {
        let delta = point[i] - query[i];
        sum += delta * delta;
    }
    sum
}

#[inline]
pub(crate) fn is_strictly_better(
    new_distance: f64,
    new_index: usize,
    old_distance: f64,
    old_index: usize,
) -> bool {
    new_distance < old_distance || (new_distance == old_distance && new_index < old_index)
}
