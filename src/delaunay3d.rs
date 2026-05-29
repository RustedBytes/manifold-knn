//! 3D Delaunay integration backed by the `delaunay` crate.
//!
//! Enable this module with the `delaunay-3d` Cargo feature. It builds the
//! successor table required by [`ManifoldKnn`](crate::ManifoldKnn) by inserting
//! points into an incremental 3D Delaunay triangulation and recording the earlier
//! Delaunay neighbors of each inserted vertex.

use std::collections::HashMap;

use delaunay::geometry::Coordinate;
use delaunay::prelude::construction::DelaunayTriangulation;
use delaunay::prelude::geometry::{AdaptiveKernel, Point};
use delaunay::prelude::tds::{Vertex, VertexKey};

use crate::{DeleteReport, Error, ManifoldKnn, Neighbor, SuccessorTable, validate_point};

type Vertex3 = Vertex<f64, usize, 3>;

/// Concrete 3D Delaunay triangulation type used by this crate.
pub type DelaunayTriangulation3 = DelaunayTriangulation<AdaptiveKernel<f64>, usize, (), 3>;

/// Incremental 3D Delaunay backend used to maintain insertion-time neighbors.
///
/// The kernel stores a mapping between the `delaunay` crate's stable vertex keys
/// and this crate's birth-order point indices. Each call to [`Self::insert`]
/// returns the earlier Delaunay neighbors of the new vertex, which are exactly
/// the edges needed to update a Manifold k-NN successor table.
#[derive(Clone, Debug)]
pub struct Delaunay3dKernel {
    triangulation: DelaunayTriangulation3,
    key_to_index: HashMap<VertexKey, usize>,
    index_to_key: Vec<Option<VertexKey>>,
    active: Vec<bool>,
}

impl Default for Delaunay3dKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl Delaunay3dKernel {
    /// Creates an empty 3D Delaunay kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            triangulation: DelaunayTriangulation3::with_empty_kernel(AdaptiveKernel::<f64>::new()),
            key_to_index: HashMap::new(),
            index_to_key: Vec::new(),
            active: Vec::new(),
        }
    }

    /// Builds a kernel and the insertion-neighbor lists for `points`.
    ///
    /// The returned `neighbors_at_insertion[j]` contains active indices `i < j`
    /// that were Delaunay-adjacent to `j` immediately after `j` was inserted.
    /// Passing these lists to [`ManifoldKnn::from_insertion_neighbors`] builds
    /// the paper's successor table.
    pub fn from_points(points: &[[f64; 3]]) -> Result<(Self, Vec<Vec<usize>>), Error> {
        let mut kernel = Self::new();
        let mut neighbors_at_insertion = Vec::with_capacity(points.len());

        for &point in points {
            let (_, neighbors) = kernel.insert(point)?;
            neighbors_at_insertion.push(neighbors);
        }

        Ok((kernel, neighbors_at_insertion))
    }

    /// Computes insertion-time 3D Delaunay neighbor lists for all points.
    ///
    /// The returned vector has one entry per input point. Entry `j` contains
    /// earlier birth indices `i < j` adjacent to `j` immediately after insertion
    /// of `j`.
    pub fn insertion_neighbors(points: &[[f64; 3]]) -> Result<Vec<Vec<usize>>, Error> {
        let (_, neighbors_at_insertion) = Self::from_points(points)?;
        Ok(neighbors_at_insertion)
    }

    /// Builds a Manifold k-NN successor table from 3D Delaunay insertions.
    pub fn successor_table(points: &[[f64; 3]]) -> Result<SuccessorTable, Error> {
        let neighbors_at_insertion = Self::insertion_neighbors(points)?;
        SuccessorTable::from_insertion_neighbors(points.len(), neighbors_at_insertion)
    }

    /// Returns the underlying `delaunay` triangulation.
    #[must_use]
    pub fn triangulation(&self) -> &DelaunayTriangulation3 {
        &self.triangulation
    }

    /// Number of points ever inserted, including inactive deleted points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index_to_key.len()
    }

    /// Returns `true` if no points have been inserted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index_to_key.is_empty()
    }

    /// Number of active points currently present in the triangulation.
    #[must_use]
    pub fn active_len(&self) -> usize {
        self.active.iter().filter(|&&is_active| is_active).count()
    }

    /// Returns whether the birth-order point index is active.
    #[must_use]
    pub fn is_active(&self, index: usize) -> bool {
        self.active.get(index).copied().unwrap_or(false)
    }

    /// Returns the Delaunay vertex key for an active point index.
    #[must_use]
    pub fn vertex_key(&self, index: usize) -> Option<VertexKey> {
        self.index_to_key.get(index).and_then(|&key| key)
    }

    /// Inserts one point and returns its birth index plus earlier Delaunay neighbors.
    ///
    /// The first four vertices bootstrap a 3D tetrahedral complex. Before the
    /// initial simplex exists, this method conservatively reports all earlier
    /// points as predecessors; this preserves the successor-table reachability
    /// invariant during the low-dimensional bootstrap prefix.
    pub fn insert(&mut self, point: [f64; 3]) -> Result<(usize, Vec<usize>), Error> {
        let birth_index = self.index_to_key.len();
        validate_point(birth_index, &point)?;

        let key = self
            .triangulation
            .insert(vertex(point, birth_index))
            .map_err(|error| Error::DelaunayKernel {
                operation: "insert",
                message: format!("{error:?}"),
            })?;

        self.key_to_index.insert(key, birth_index);
        self.index_to_key.push(Some(key));
        self.active.push(true);

        let mut prior_neighbors = self.prior_neighbors_for_inserted_key(key, birth_index)?;
        if birth_index < 4 {
            prior_neighbors.extend(0..birth_index);
        }
        prior_neighbors.sort_unstable();
        prior_neighbors.dedup();

        Ok((birth_index, prior_neighbors))
    }

    /// Returns active Delaunay graph neighbors for the point at `index`.
    pub fn neighbors(&self, index: usize) -> Result<Vec<usize>, Error> {
        let key = self.ensure_active_key(index)?;
        self.incident_neighbors_from_key(key)
    }

    /// Returns all active Delaunay graph edges as birth-index pairs `(min, max)`.
    pub fn current_edges(&self) -> Result<Vec<(usize, usize)>, Error> {
        let mut edges = Vec::new();

        for edge in self.triangulation.edges() {
            let (a, b) = edge.endpoints();
            let i = self.index_for_key(a)?;
            let j = self.index_for_key(b)?;

            if self.is_active(i) && self.is_active(j) && i != j {
                edges.push(if i < j { (i, j) } else { (j, i) });
            }
        }

        edges.sort_unstable();
        edges.dedup();
        Ok(edges)
    }

    /// Removes an active vertex and returns the active Delaunay graph after removal.
    ///
    /// The returned edge list can be passed to
    /// [`ManifoldKnn::delete_with_new_successors`]. This is a conservative update:
    /// it inserts all currently active Delaunay edges into the successor table, not
    /// only the local cavity edges created by deletion. That preserves correctness
    /// but can increase table density after many deletions. The method clones the
    /// backend before removal so it can roll back this wrapper if post-removal
    /// Delaunay repair fails.
    pub fn remove(&mut self, index: usize) -> Result<Vec<(usize, usize)>, Error> {
        let key = self.ensure_active_key(index)?;
        let rollback = self.clone();

        self.triangulation
            .remove_vertex(key)
            .map_err(|error| Error::DelaunayKernel {
                operation: "remove_vertex",
                message: format!("{error:?}"),
            })?;

        if let Err(error) = self.triangulation.repair_delaunay_with_flips() {
            *self = rollback;
            return Err(Error::DelaunayKernel {
                operation: "repair_delaunay_with_flips",
                message: format!("{error:?}"),
            });
        }

        self.key_to_index.remove(&key);
        self.index_to_key[index] = None;
        self.active[index] = false;

        self.current_edges()
    }

    fn prior_neighbors_for_inserted_key(
        &self,
        key: VertexKey,
        birth_index: usize,
    ) -> Result<Vec<usize>, Error> {
        let mut neighbors = self.incident_neighbors_from_key(key)?;
        neighbors.retain(|&index| index < birth_index);
        Ok(neighbors)
    }

    fn incident_neighbors_from_key(&self, key: VertexKey) -> Result<Vec<usize>, Error> {
        let mut neighbors = Vec::new();

        for edge in self.triangulation.incident_edges(key) {
            let (a, b) = edge.endpoints();
            let other = if a == key {
                b
            } else if b == key {
                a
            } else {
                return Err(Error::DelaunayInvariant {
                    message: "incident edge did not contain the requested vertex".to_owned(),
                });
            };

            let index = self.index_for_key(other)?;
            if self.is_active(index) {
                neighbors.push(index);
            }
        }

        neighbors.sort_unstable();
        neighbors.dedup();
        Ok(neighbors)
    }

    fn ensure_active_key(&self, index: usize) -> Result<VertexKey, Error> {
        if index >= self.index_to_key.len() {
            return Err(Error::InvalidIndex {
                index,
                len: self.index_to_key.len(),
            });
        }
        if !self.active[index] {
            return Err(Error::InactivePoint { index });
        }
        self.index_to_key[index].ok_or(Error::InactivePoint { index })
    }

    fn index_for_key(&self, key: VertexKey) -> Result<usize, Error> {
        self.key_to_index
            .get(&key)
            .copied()
            .ok_or_else(|| Error::DelaunayInvariant {
                message: "Delaunay edge endpoint is missing from the key/index map".to_owned(),
            })
    }
}

/// A synchronized 3D Manifold k-NN index and incremental Delaunay backend.
///
/// Use this wrapper when you want the crate to maintain both the Delaunay graph
/// and the successor table for dynamic 3D point sets.
#[derive(Clone, Debug)]
pub struct DelaunayManifoldKnn3 {
    index: ManifoldKnn<3>,
    kernel: Delaunay3dKernel,
}

impl Default for DelaunayManifoldKnn3 {
    fn default() -> Self {
        Self::new()
    }
}

impl DelaunayManifoldKnn3 {
    /// Creates an empty synchronized Delaunay/Manifold k-NN index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            index: ManifoldKnn::new(Vec::new(), SuccessorTable::empty(0))
                .expect("empty successor table is valid"),
            kernel: Delaunay3dKernel::new(),
        }
    }

    /// Builds a synchronized index from 3D points in birth order.
    pub fn from_points(points: Vec<[f64; 3]>) -> Result<Self, Error> {
        let (kernel, neighbors_at_insertion) = Delaunay3dKernel::from_points(&points)?;
        let index = ManifoldKnn::from_insertion_neighbors(points, neighbors_at_insertion)?;
        Ok(Self { index, kernel })
    }

    /// Returns the Manifold k-NN query index.
    #[must_use]
    pub fn index(&self) -> &ManifoldKnn<3> {
        &self.index
    }

    /// Returns the Delaunay backend.
    #[must_use]
    pub fn kernel(&self) -> &Delaunay3dKernel {
        &self.kernel
    }

    /// Consumes the wrapper and returns the query index.
    #[must_use]
    pub fn into_index(self) -> ManifoldKnn<3> {
        self.index
    }

    /// Returns all points in birth order.
    #[must_use]
    pub fn points(&self) -> &[[f64; 3]] {
        self.index.points()
    }

    /// Returns the successor table.
    #[must_use]
    pub fn successors(&self) -> &SuccessorTable {
        self.index.successors()
    }

    /// Number of stored points, including inactive deleted points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Returns `true` if no points have been inserted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Number of active points.
    #[must_use]
    pub fn active_len(&self) -> usize {
        self.index.active_len()
    }

    /// Returns whether `index` is active.
    #[must_use]
    pub fn is_active(&self, index: usize) -> bool {
        self.index.is_active(index)
    }

    /// Inserts one 3D point and updates both the Delaunay backend and successor table.
    pub fn insert(&mut self, point: [f64; 3]) -> Result<usize, Error> {
        let (delaunay_index, prior_neighbors) = self.kernel.insert(point)?;
        let index = self.index.insert_with_neighbors(point, prior_neighbors)?;

        debug_assert_eq!(delaunay_index, index);
        Ok(index)
    }

    /// Deletes an active point from both the Delaunay backend and successor table.
    ///
    /// The Delaunay backend returns all current active graph edges after removal;
    /// the successor table inserts those edges conservatively. This is exact but
    /// may become denser than a minimal local-cavity update.
    pub fn delete(&mut self, index: usize) -> Result<DeleteReport, Error> {
        let current_edges = self.kernel.remove(index)?;
        self.index.delete_with_new_successors(index, current_edges)
    }

    /// Returns the nearest active point in the full index.
    pub fn nearest(&self, query: &[f64; 3]) -> Result<Option<Neighbor>, Error> {
        self.index.nearest(query)
    }

    /// Returns up to `k` nearest active points in the full index.
    pub fn knn(&self, query: &[f64; 3], k: usize) -> Result<Vec<Neighbor>, Error> {
        self.index.knn(query, k)
    }

    /// Returns up to `k` nearest active points whose birth index is `< prefix_len`.
    pub fn knn_prefix(
        &self,
        query: &[f64; 3],
        k: usize,
        prefix_len: usize,
    ) -> Result<Vec<Neighbor>, Error> {
        self.index.knn_prefix(query, k, prefix_len)
    }
}

impl SuccessorTable {
    /// Builds a successor table from incremental 3D Delaunay insertions.
    ///
    /// This is a convenience wrapper around [`Delaunay3dKernel::successor_table`].
    pub fn from_delaunay3d(points: &[[f64; 3]]) -> Result<Self, Error> {
        Delaunay3dKernel::successor_table(points)
    }
}

impl ManifoldKnn<3> {
    /// Builds a 3D Manifold k-NN index directly from Delaunay insertion neighbors.
    ///
    /// This is the batch convenience API for the optional `delaunay-3d` feature.
    /// Use [`DelaunayManifoldKnn3`] when you also need dynamic insertions and
    /// deletions to keep the Delaunay backend synchronized.
    pub fn from_delaunay(points: Vec<[f64; 3]>) -> Result<Self, Error> {
        let (_, neighbors_at_insertion) = Delaunay3dKernel::from_points(&points)?;
        Self::from_insertion_neighbors(points, neighbors_at_insertion)
    }
}

fn vertex(point: [f64; 3], birth_index: usize) -> Vertex3 {
    Vertex::from_point_with_data(Point::new(point), birth_index)
}
