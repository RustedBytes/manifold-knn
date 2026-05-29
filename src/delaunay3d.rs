//! 3D Delaunay integration backed by a custom optimized triangulation.
//!
//! Enable this module with the `delaunay-3d` Cargo feature. It builds the
//! successor table required by [`ManifoldKnn`](crate::ManifoldKnn) by inserting
//! points into an incremental 3D Delaunay triangulation and recording the earlier
//! Delaunay neighbors of each inserted vertex.

use std::collections::{HashMap, HashSet};

use crate::{DeleteReport, Error, ManifoldKnn, Neighbor, SuccessorTable, validate_point};

/// A key identifying a vertex in the Delaunay triangulation.
pub type VertexKey = usize;

/// Tetrahedron representation inside the custom triangulation.
#[derive(Clone, Copy, Debug)]
pub struct Tetrahedron {
    /// The indices of the 4 vertices forming the tetrahedron.
    pub vertices: [usize; 4],
    /// The indices of the 4 neighbor tetrahedra opposite to each vertex, if any.
    pub neighbors: [Option<usize>; 4],
}

impl Tetrahedron {
    /// Returns `true` if this tetrahedron has been deleted.
    #[inline]
    pub fn is_deleted(&self) -> bool {
        self.vertices[0] == usize::MAX
    }
}

/// KD-tree node for fast nearest neighbor point location.
#[derive(Clone, Copy, Debug)]
pub struct KdNode {
    /// The point index in the triangulation.
    pub point_idx: usize,
    /// Left child index.
    pub left: Option<usize>,
    /// Right child index.
    pub right: Option<usize>,
}

/// Incremental 3D KD-tree.
#[derive(Clone, Debug, Default)]
pub struct KdTree {
    /// The nodes in the KD-tree.
    pub nodes: Vec<KdNode>,
    /// The root node index, if any.
    pub root: Option<usize>,
}

impl KdTree {
    /// Inserts a point into the KD-tree.
    pub fn insert(&mut self, points: &[[f64; 3]], point_idx: usize) {
        let new_node_idx = self.nodes.len();
        self.nodes.push(KdNode {
            point_idx,
            left: None,
            right: None,
        });

        if self.root.is_none() {
            self.root = Some(new_node_idx);
            return;
        }

        let mut curr = self.root.unwrap();
        let mut depth = 0;
        let p = points[point_idx];

        loop {
            let axis = depth % 3;
            // Use index to avoid borrow checker issues with mutability
            let node_p = points[self.nodes[curr].point_idx];

            if p[axis] < node_p[axis] {
                if let Some(left) = self.nodes[curr].left {
                    curr = left;
                } else {
                    self.nodes[curr].left = Some(new_node_idx);
                    break;
                }
            } else {
                if let Some(right) = self.nodes[curr].right {
                    curr = right;
                } else {
                    self.nodes[curr].right = Some(new_node_idx);
                    break;
                }
            }
            depth += 1;
        }
    }

    /// Finds the nearest neighbor to `query` point in the KD-tree.
    pub fn find_nearest(&self, points: &[[f64; 3]], query: [f64; 3]) -> Option<usize> {
        let mut best_idx = None;
        let mut best_dist_sq = f64::INFINITY;
        self.nearest_recurse(
            self.root,
            0,
            points,
            query,
            &mut best_idx,
            &mut best_dist_sq,
        );
        best_idx
    }

    fn nearest_recurse(
        &self,
        node_idx: Option<usize>,
        depth: usize,
        points: &[[f64; 3]],
        query: [f64; 3],
        best_idx: &mut Option<usize>,
        best_dist_sq: &mut f64,
    ) {
        let Some(idx) = node_idx else {
            return;
        };
        let node = &self.nodes[idx];
        let p = points[node.point_idx];

        let dist_sq =
            (p[0] - query[0]).powi(2) + (p[1] - query[1]).powi(2) + (p[2] - query[2]).powi(2);
        if dist_sq < *best_dist_sq {
            *best_dist_sq = dist_sq;
            *best_idx = Some(node.point_idx);
        }

        let axis = depth % 3;
        let diff = query[axis] - p[axis];

        let (first, second) = if diff < 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        self.nearest_recurse(first, depth + 1, points, query, best_idx, best_dist_sq);

        if diff.powi(2) < *best_dist_sq {
            self.nearest_recurse(second, depth + 1, points, query, best_idx, best_dist_sq);
        }
    }
}

/// Custom 3D Delaunay triangulation backend using Shewchuk's robust predicates.
#[derive(Clone, Debug)]
pub struct CustomDelaunay3d {
    /// Coordinates of inserted points.
    pub points: Vec<[f64; 3]>,
    /// Coordinates of the super-tetrahedron vertices.
    pub super_points: [[f64; 3]; 4],
    /// List of all tetrahedra.
    pub tetrahedra: Vec<Tetrahedron>,
    /// Mapping from vertex index to list of active tetrahedra containing it.
    pub vertex_to_tets: Vec<Vec<usize>>,
    /// KD-tree for fast nearest vertex location.
    pub kd_tree: KdTree,
    /// Number of points inserted into the triangulation.
    pub num_points: usize,
    /// Radius of the super-tetrahedron.
    pub super_r: f64,
    /// Center of the super-tetrahedron.
    pub center: [f64; 3],
}

#[inline]
fn to_coord(p: [f64; 3]) -> robust::Coord3D<f64> {
    robust::Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

impl CustomDelaunay3d {
    /// Creates a new CustomDelaunay3d triangulation with a given super-radius and center.
    pub fn new(super_r: f64, center: [f64; 3]) -> Self {
        let cx = center[0];
        let cy = center[1];
        let cz = center[2];
        let r = super_r;
        let super_points = [
            [cx, cy + 3.0 * r, cz],
            [cx - 2.0 * r * 1.732, cy - r, cz - r * 1.414],
            [cx + 2.0 * r * 1.732, cy - r, cz - r * 1.414],
            [cx, cy - r, cz + 3.0 * r * 1.414],
        ];

        let mut tets = Vec::new();
        let u = usize::MAX - 3;
        let v = usize::MAX - 2;
        let w = usize::MAX - 1;
        let z = usize::MAX;

        let mut vertices = [u, v, w, z];
        let o = robust::orient3d(
            to_coord(super_points[0]),
            to_coord(super_points[1]),
            to_coord(super_points[2]),
            to_coord(super_points[3]),
        );
        if o < 0.0 {
            vertices.swap(0, 1);
        }

        tets.push(Tetrahedron {
            vertices,
            neighbors: [None; 4],
        });

        Self {
            points: Vec::new(),
            super_points,
            tetrahedra: tets,
            vertex_to_tets: Vec::new(),
            kd_tree: KdTree::default(),
            num_points: 0,
            super_r,
            center,
        }
    }

    /// Creates a default CustomDelaunay3d triangulation with super-radius 1e5 at origin.
    pub fn default() -> Self {
        Self::new(1e5, [0.0, 0.0, 0.0])
    }

    #[inline]
    fn get_coord(&self, idx: usize) -> [f64; 3] {
        if idx >= usize::MAX - 3 {
            self.super_points[idx - (usize::MAX - 3)]
        } else {
            self.points[idx]
        }
    }

    fn locate_cell(&self, start_tet: usize, p: [f64; 3]) -> usize {
        let mut curr = start_tet;
        let mut visited = HashSet::new();
        visited.insert(curr);

        loop {
            let tet = &self.tetrahedra[curr];
            let v = tet.vertices;

            let faces = [
                (0, [v[1], v[2], v[3]], v[0]),
                (1, [v[0], v[3], v[2]], v[1]),
                (2, [v[0], v[1], v[3]], v[2]),
                (3, [v[0], v[2], v[1]], v[3]),
            ];

            let mut walked = false;
            for &(face_idx, face_verts, opp) in &faces {
                let pa = self.get_coord(face_verts[0]);
                let pb = self.get_coord(face_verts[1]);
                let pc = self.get_coord(face_verts[2]);
                let popp = self.get_coord(opp);

                let val_opp =
                    robust::orient3d(to_coord(pa), to_coord(pb), to_coord(pc), to_coord(popp));
                let val_p = robust::orient3d(to_coord(pa), to_coord(pb), to_coord(pc), to_coord(p));

                if val_p * val_opp < 0.0 {
                    if let Some(next_tet) = tet.neighbors[face_idx] {
                        if !visited.contains(&next_tet) {
                            visited.insert(next_tet);
                            curr = next_tet;
                            walked = true;
                            break;
                        }
                    }
                }
            }

            if !walked {
                return curr;
            }
        }
    }

    fn in_circumsphere(&self, tet_idx: usize, p: [f64; 3]) -> bool {
        let tet = &self.tetrahedra[tet_idx];
        let v = tet.vertices;
        let pa = self.get_coord(v[0]);
        let pb = self.get_coord(v[1]);
        let pc = self.get_coord(v[2]);
        let pd = self.get_coord(v[3]);

        robust::insphere(
            to_coord(pa),
            to_coord(pb),
            to_coord(pc),
            to_coord(pd),
            to_coord(p),
        ) > 0.0
    }

    /// Inserts a point into the 3D triangulation and returns its index.
    pub fn insert(&mut self, p: [f64; 3]) -> usize {
        let p_idx = self.points.len();
        self.points.push(p);
        self.vertex_to_tets.push(Vec::new());

        // Check if there is an identical/coincident point already inserted
        if let Some(nearest_v) = self.kd_tree.find_nearest(&self.points[..p_idx], p) {
            let np = self.points[nearest_v];
            let dist_sq = (np[0] - p[0]).powi(2) + (np[1] - p[1]).powi(2) + (np[2] - p[2]).powi(2);
            if dist_sq < 1e-16 {
                // Duplicate point. Map it but do not insert into the geometry.
                self.num_points += 1;
                return p_idx;
            }
        }

        self.insert_at(p_idx, p);
        p_idx
    }

    fn insert_at(&mut self, p_idx: usize, p: [f64; 3]) {
        let t_start = if let Some(nearest_v) = self.kd_tree.find_nearest(&self.points[..p_idx], p) {
            let mut found = None;
            for &t_idx in &self.vertex_to_tets[nearest_v] {
                if !self.tetrahedra[t_idx].is_deleted() {
                    found = Some(t_idx);
                    break;
                }
            }
            found.unwrap_or(0)
        } else {
            0
        };

        let t_containing = self.locate_cell(t_start, p);

        let mut cavity = Vec::new();
        let mut visited = vec![false; self.tetrahedra.len()];
        let mut queue = vec![t_containing];
        visited[t_containing] = true;

        let mut q_head = 0;
        while q_head < queue.len() {
            let t_idx = queue[q_head];
            q_head += 1;

            if self.in_circumsphere(t_idx, p) {
                cavity.push(t_idx);
                for &neighbor in &self.tetrahedra[t_idx].neighbors {
                    if let Some(n_idx) = neighbor {
                        if !visited[n_idx] {
                            visited[n_idx] = true;
                            queue.push(n_idx);
                        }
                    }
                }
            }
        }

        let mut boundary = Vec::new();
        let mut cavity_set = vec![false; self.tetrahedra.len()];
        for &t_idx in &cavity {
            cavity_set[t_idx] = true;
        }

        for &t_idx in &cavity {
            let tet = &self.tetrahedra[t_idx];
            for face_idx in 0..4 {
                let neighbor = tet.neighbors[face_idx];
                let is_boundary = match neighbor {
                    None => true,
                    Some(n_idx) => !cavity_set[n_idx],
                };
                if is_boundary {
                    boundary.push((t_idx, face_idx));
                }
            }
        }

        let new_tets_start_idx = self.tetrahedra.len();
        let mut new_tets = Vec::with_capacity(boundary.len());
        let mut edge_to_face = HashMap::new();

        for (b_idx, &(t_idx, face_idx)) in boundary.iter().enumerate() {
            let tet = &self.tetrahedra[t_idx];
            let v = tet.vertices;

            let face_verts = match face_idx {
                0 => [v[1], v[2], v[3]],
                1 => [v[0], v[3], v[2]],
                2 => [v[0], v[1], v[3]],
                3 => [v[0], v[2], v[1]],
                _ => unreachable!(),
            };

            let new_tet_idx = new_tets_start_idx + b_idx;
            let mut new_vertices = [face_verts[0], face_verts[1], face_verts[2], p_idx];

            let mut neighbors = [None; 4];
            neighbors[3] = tet.neighbors[face_idx];

            let mut edges = [
                (face_verts[1], face_verts[2]),
                (face_verts[0], face_verts[2]),
                (face_verts[0], face_verts[1]),
            ];

            let o = robust::orient3d(
                to_coord(self.get_coord(face_verts[0])),
                to_coord(self.get_coord(face_verts[1])),
                to_coord(self.get_coord(face_verts[2])),
                to_coord(p),
            );
            if o < 0.0 {
                new_vertices.swap(0, 1);
                neighbors.swap(0, 1);
                edges.swap(0, 1);
            }

            new_tets.push(Tetrahedron {
                vertices: new_vertices,
                neighbors,
            });

            if let Some(n_idx) = tet.neighbors[face_idx] {
                let n_tet = &mut self.tetrahedra[n_idx];
                for n_face_idx in 0..4 {
                    if n_tet.neighbors[n_face_idx] == Some(t_idx) {
                        n_tet.neighbors[n_face_idx] = Some(new_tet_idx);
                        break;
                    }
                }
            }

            for face_sub_idx in 0..3 {
                let edge = edges[face_sub_idx];
                let sorted_edge = if edge.0 < edge.1 {
                    (edge.0, edge.1)
                } else {
                    (edge.1, edge.0)
                };

                if let Some(&(other_tet_idx, other_face_idx)) = edge_to_face.get(&sorted_edge) {
                    new_tets[b_idx].neighbors[face_sub_idx] = Some(other_tet_idx);
                    if other_tet_idx < new_tets_start_idx {
                        self.tetrahedra[other_tet_idx].neighbors[other_face_idx] =
                            Some(new_tet_idx);
                    } else {
                        new_tets[other_tet_idx - new_tets_start_idx].neighbors[other_face_idx] =
                            Some(new_tet_idx);
                    }
                } else {
                    edge_to_face.insert(sorted_edge, (new_tet_idx, face_sub_idx));
                }
            }
        }

        for &t_idx in &cavity {
            self.tetrahedra[t_idx].vertices = [usize::MAX; 4];
            self.tetrahedra[t_idx].neighbors = [None; 4];
        }

        for (b_idx, new_tet) in new_tets.into_iter().enumerate() {
            let new_tet_idx = new_tets_start_idx + b_idx;
            for &v_idx in &new_tet.vertices {
                if v_idx < usize::MAX - 3 {
                    self.vertex_to_tets[v_idx].push(new_tet_idx);
                }
            }
            self.tetrahedra.push(new_tet);
        }

        self.kd_tree.insert(&self.points, p_idx);
        self.num_points += 1;
    }

    /// Returns all Delaunay neighbors of the vertex with index `v`.
    pub fn incident_neighbors(&self, v: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();
        if v >= self.vertex_to_tets.len() {
            return neighbors;
        }
        for &tet_idx in &self.vertex_to_tets[v] {
            let tet = &self.tetrahedra[tet_idx];
            if tet.is_deleted() {
                continue;
            }
            for &u in &tet.vertices {
                if u != v && u < self.points.len() {
                    neighbors.push(u);
                }
            }
        }
        neighbors.sort_unstable();
        neighbors.dedup();
        neighbors
    }
}

/// Concrete 3D Delaunay triangulation type used by this crate.
pub type DelaunayTriangulation3 = CustomDelaunay3d;

/// Incremental 3D Delaunay backend used to maintain insertion-time neighbors.
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
            triangulation: CustomDelaunay3d::default(),
            key_to_index: HashMap::new(),
            index_to_key: Vec::new(),
            active: Vec::new(),
        }
    }

    /// Builds a kernel and the insertion-neighbor lists for `points`.
    pub fn from_points(points: &[[f64; 3]]) -> Result<(Self, Vec<Vec<usize>>), Error> {
        if points.is_empty() {
            return Ok((Self::new(), Vec::new()));
        }

        // Bounding box computation for custom super-tetrahedron sizing
        let mut xmin = points[0][0];
        let mut xmax = points[0][0];
        let mut ymin = points[0][1];
        let mut ymax = points[0][1];
        let mut zmin = points[0][2];
        let mut zmax = points[0][2];
        for &p in points {
            xmin = xmin.min(p[0]);
            xmax = xmax.max(p[0]);
            ymin = ymin.min(p[1]);
            ymax = ymax.max(p[1]);
            zmin = zmin.min(p[2]);
            zmax = zmax.max(p[2]);
        }
        let cx = (xmin + xmax) / 2.0;
        let cy = (ymin + ymax) / 2.0;
        let cz = (zmin + zmax) / 2.0;
        let dx = xmax - xmin;
        let dy = ymax - ymin;
        let dz = zmax - zmin;
        let max_span = dx.max(dy).max(dz).max(1.0);
        let super_r = max_span * 100.0;

        let mut kernel = Self {
            triangulation: CustomDelaunay3d::new(super_r, [cx, cy, cz]),
            key_to_index: HashMap::new(),
            index_to_key: Vec::new(),
            active: Vec::new(),
        };

        let mut neighbors_at_insertion = Vec::with_capacity(points.len());
        for &point in points {
            let (_, neighbors) = kernel.insert(point)?;
            neighbors_at_insertion.push(neighbors);
        }

        Ok((kernel, neighbors_at_insertion))
    }

    /// Computes insertion-time 3D Delaunay neighbor lists for all points.
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
    #[inline]
    pub fn len(&self) -> usize {
        self.index_to_key.len()
    }

    /// Returns `true` if no points have been inserted.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.index_to_key.is_empty()
    }

    /// Number of active points currently present in the triangulation.
    #[must_use]
    #[inline]
    pub fn active_len(&self) -> usize {
        self.active.iter().filter(|&&is_active| is_active).count()
    }

    /// Returns whether the birth-order point index is active.
    #[must_use]
    #[inline]
    pub fn is_active(&self, index: usize) -> bool {
        self.active.get(index).copied().unwrap_or(false)
    }

    /// Returns the Delaunay vertex key for an active point index.
    #[must_use]
    #[inline]
    pub fn vertex_key(&self, index: usize) -> Option<VertexKey> {
        self.index_to_key.get(index).and_then(|&key| key)
    }

    /// Inserts one point and returns its birth index plus earlier Delaunay neighbors.
    pub fn insert(&mut self, point: [f64; 3]) -> Result<(usize, Vec<usize>), Error> {
        let birth_index = self.index_to_key.len();
        validate_point(birth_index, &point)?;

        if birth_index == 0 {
            self.triangulation = CustomDelaunay3d::new(1e5, point);
        }

        let key = self.triangulation.insert(point);

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

        for tet in &self.triangulation.tetrahedra {
            if tet.is_deleted() {
                continue;
            }
            let v = tet.vertices;
            let tet_edges = [
                (v[0], v[1]),
                (v[0], v[2]),
                (v[0], v[3]),
                (v[1], v[2]),
                (v[1], v[3]),
                (v[2], v[3]),
            ];
            for &(a, b) in &tet_edges {
                if a < self.triangulation.points.len() && b < self.triangulation.points.len() {
                    if self.is_active(a) && self.is_active(b) && a != b {
                        edges.push(if a < b { (a, b) } else { (b, a) });
                    }
                }
            }
        }

        edges.sort_unstable();
        edges.dedup();
        Ok(edges)
    }

    /// Removes an active vertex and returns the active Delaunay graph after removal.
    pub fn remove(&mut self, index: usize) -> Result<Vec<(usize, usize)>, Error> {
        let key = self.ensure_active_key(index)?;

        self.key_to_index.remove(&key);
        self.index_to_key[index] = None;
        self.active[index] = false;

        // Rebuild triangulation from scratch with remaining active points
        let super_r = self.triangulation.super_r;
        let center = self.triangulation.center;
        let mut new_triangulation = CustomDelaunay3d::new(super_r, center);

        new_triangulation.points = self.triangulation.points.clone();
        new_triangulation.vertex_to_tets = vec![Vec::new(); new_triangulation.points.len()];

        for i in 0..new_triangulation.points.len() {
            if self.is_active(i) {
                let p = new_triangulation.points[i];
                new_triangulation.insert_at(i, p);
            }
        }

        self.triangulation = new_triangulation;
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
        let mut neighbors = self.triangulation.incident_neighbors(key);
        neighbors.retain(|&idx| self.is_active(idx));
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
}

/// A synchronized 3D Manifold k-NN index and incremental Delaunay backend.
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
    #[inline]
    pub fn index(&self) -> &ManifoldKnn<3> {
        &self.index
    }

    /// Returns the Delaunay backend.
    #[must_use]
    #[inline]
    pub fn kernel(&self) -> &Delaunay3dKernel {
        &self.kernel
    }

    /// Consumes the wrapper and returns the query index.
    #[must_use]
    #[inline]
    pub fn into_index(self) -> ManifoldKnn<3> {
        self.index
    }

    /// Returns all points in birth order.
    #[must_use]
    #[inline]
    pub fn points(&self) -> &[[f64; 3]] {
        self.index.points()
    }

    /// Returns the successor table.
    #[must_use]
    #[inline]
    pub fn successors(&self) -> &SuccessorTable {
        self.index.successors()
    }

    /// Number of stored points, including inactive deleted points.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Returns `true` if no points have been inserted.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Number of active points.
    #[must_use]
    #[inline]
    pub fn active_len(&self) -> usize {
        self.index.active_len()
    }

    /// Returns whether `index` is active.
    #[must_use]
    #[inline]
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
    pub fn delete(&mut self, index: usize) -> Result<DeleteReport, Error> {
        let current_edges = self.kernel.remove(index)?;
        self.index.delete_with_new_successors(index, current_edges)
    }

    /// Returns the nearest active point in the full index.
    pub fn nearest(&self, query: &[f64; 3]) -> Result<Option<Neighbor>, Error> {
        self.index.nearest(query)
    }

    /// Returns the nearest active point in the full index using a workspace to avoid allocations.
    pub fn nearest_with_workspace(
        &self,
        query: &[f64; 3],
        workspace: &mut crate::QueryWorkspace,
    ) -> Result<Option<Neighbor>, Error> {
        self.index.nearest_with_workspace(query, workspace)
    }

    /// Returns up to `k` nearest active points in the full index.
    pub fn knn(&self, query: &[f64; 3], k: usize) -> Result<Vec<Neighbor>, Error> {
        self.index.knn(query, k)
    }

    /// Returns up to `k` nearest active points in the full index using a workspace.
    pub fn knn_with_workspace<'a>(
        &self,
        query: &[f64; 3],
        k: usize,
        workspace: &'a mut crate::QueryWorkspace,
    ) -> Result<&'a [Neighbor], Error> {
        self.index.knn_with_workspace(query, k, workspace)
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

    /// Returns up to `k` nearest active points whose birth index is `< prefix_len` using a workspace.
    pub fn knn_prefix_with_workspace<'a>(
        &self,
        query: &[f64; 3],
        k: usize,
        prefix_len: usize,
        workspace: &'a mut crate::QueryWorkspace,
    ) -> Result<&'a [Neighbor], Error> {
        self.index
            .knn_prefix_with_workspace(query, k, prefix_len, workspace)
    }
}

impl SuccessorTable {
    /// Builds a successor table from incremental 3D Delaunay insertions.
    pub fn from_delaunay3d(points: &[[f64; 3]]) -> Result<Self, Error> {
        Delaunay3dKernel::successor_table(points)
    }
}

impl ManifoldKnn<3> {
    /// Builds a 3D Manifold k-NN index directly from Delaunay insertion neighbors.
    pub fn from_delaunay(points: Vec<[f64; 3]>) -> Result<Self, Error> {
        let (_, neighbors_at_insertion) = Delaunay3dKernel::from_points(&points)?;
        Self::from_insertion_neighbors(points, neighbors_at_insertion)
    }
}
