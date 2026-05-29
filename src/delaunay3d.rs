//! 3D Delaunay integration backed by a custom optimized triangulation.
//!
//! Enable this module with the `delaunay-3d` Cargo feature. It builds the
//! successor table required by [`ManifoldKnn`] by inserting
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

/// Computes a 3D Z-order (Morton) key for a point.
///
/// Coordinates are quantized to 21 bits per dimension to fit a 63-bit key.
pub fn z_order_key(p: [f64; 3], min_val: [f64; 3], max_val: [f64; 3]) -> u64 {
    let quantize = |val: f64, min: f64, max: f64| -> u32 {
        let span = max - min;
        if span <= 1e-9 {
            return 0;
        }
        let norm = ((val - min) / span).clamp(0.0, 1.0);
        (norm * ((1 << 21) - 1) as f64) as u32
    };

    let x = quantize(p[0], min_val[0], max_val[0]);
    let y = quantize(p[1], min_val[1], max_val[1]);
    let z = quantize(p[2], min_val[2], max_val[2]);

    let mut key = 0u64;
    for i in 0..21 {
        key |= (((x >> i) & 1) as u64) << (3 * i);
        key |= (((y >> i) & 1) as u64) << (3 * i + 1);
        key |= (((z >> i) & 1) as u64) << (3 * i + 2);
    }
    key
}

/// Sorts a slice of points in-place using a Biased Randomized Incremental Order (BRIO)
/// with Z-order (Morton) keys applied within each bucket.
///
/// Returns the permutation mapping: `permutation[orig_idx] = new_idx`.
pub fn sort_brio_spatial(points: &mut [[f64; 3]]) -> Vec<usize> {
    if points.is_empty() {
        return Vec::new();
    }

    // 1. Compute bounding box
    let mut min_val = points[0];
    let mut max_val = points[0];
    for &p in points.iter() {
        for d in 0..3 {
            min_val[d] = min_val[d].min(p[d]);
            max_val[d] = max_val[d].max(p[d]);
        }
    }

    // 2. Assign level/bucket and compute Z-order key
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next_random = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mantissa = state >> 12;
        (mantissa as f64) / ((1_u64 << 52) as f64)
    };

    struct PointInfo {
        orig_idx: usize,
        level: usize,
        spatial_key: u64,
    }

    let mut infos: Vec<PointInfo> = points
        .iter()
        .enumerate()
        .map(|(idx, &p)| {
            let u = next_random().max(1e-15);
            let mut level = (-u.log2()).floor() as usize;
            if level > 12 {
                level = 12;
            }

            let key = z_order_key(p, min_val, max_val);

            PointInfo {
                orig_idx: idx,
                level,
                spatial_key: key,
            }
        })
        .collect();

    // 3. Sort by level descending (highest level first), then by spatial key
    infos.sort_unstable_by(|a, b| {
        let cmp = b.level.cmp(&a.level);
        if cmp == std::cmp::Ordering::Equal {
            a.spatial_key.cmp(&b.spatial_key)
        } else {
            cmp
        }
    });

    // 4. Rearrange points
    let reordered: Vec<[f64; 3]> = infos.iter().map(|info| points[info.orig_idx]).collect();
    points.copy_from_slice(&reordered);

    // 5. Build permutation map: new_idx = permutation[orig_idx]
    let mut permutation = vec![0; points.len()];
    for (new_idx, info) in infos.iter().enumerate() {
        permutation[info.orig_idx] = new_idx;
    }
    permutation
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
    /// Index of the last inserted active tetrahedron (for starting visibility walk).
    pub last_tet: usize,
    /// Number of points inserted into the triangulation.
    pub num_points: usize,
    /// Radius of the super-tetrahedron.
    pub super_r: f64,
    /// Center of the super-tetrahedron.
    pub center: [f64; 3],
    /// Reusable visited buffer to avoid allocations during cavity searches.
    visited: Vec<u32>,
    /// Generation identifier for the visited buffer.
    current_visit_id: u32,
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
            last_tet: 0,
            num_points: 0,
            super_r,
            center,
            visited: vec![0; 1],
            current_visit_id: 1,
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

        // Fast path: stack-allocated visited list to avoid heap allocations
        let mut visited_stack = [usize::MAX; 16];
        visited_stack[0] = curr;
        let mut visited_count = 1;
        let mut fallback_set: Option<HashSet<usize>> = None;

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

                let val_opp = robust::orient3d(to_coord(pa), to_coord(pb), to_coord(pc), to_coord(popp));
                let val_p = robust::orient3d(to_coord(pa), to_coord(pb), to_coord(pc), to_coord(p));

                if val_p * val_opp < 0.0 {
                    if let Some(next_tet) = tet.neighbors[face_idx] {
                        let already_visited = if let Some(ref set) = fallback_set {
                            set.contains(&next_tet)
                        } else {
                            visited_stack[..visited_count].contains(&next_tet)
                        };

                        if !already_visited {
                            if let Some(ref mut set) = fallback_set {
                                set.insert(next_tet);
                            } else if visited_count < 16 {
                                visited_stack[visited_count] = next_tet;
                                visited_count += 1;
                            } else {
                                let mut set = HashSet::with_capacity(32);
                                for &t in &visited_stack[..visited_count] {
                                    set.insert(t);
                                }
                                set.insert(next_tet);
                                fallback_set = Some(set);
                            }
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

        robust::insphere(to_coord(pa), to_coord(pb), to_coord(pc), to_coord(pd), to_coord(p)) > 0.0
    }

    /// Inserts a point into the 3D triangulation and returns its index.
    pub fn insert(&mut self, p: [f64; 3]) -> usize {
        let p_idx = self.points.len();
        self.points.push(p);
        self.vertex_to_tets.push(Vec::new());

        let t_containing = self.locate_cell(self.last_tet, p);
        for &v_idx in &self.tetrahedra[t_containing].vertices {
            if v_idx < usize::MAX - 3 {
                let np = self.points[v_idx];
                let dist_sq = (np[0] - p[0]).powi(2) + (np[1] - p[1]).powi(2) + (np[2] - p[2]).powi(2);
                if dist_sq < 1e-16 {
                    self.num_points += 1;
                    return p_idx;
                }
            }
        }

        self.insert_at(p_idx, p, t_containing);
        p_idx
    }

    fn insert_at(&mut self, p_idx: usize, p: [f64; 3], t_containing: usize) {
        // Increment visit generation by 2.
        // self.current_visit_id represents "visited/queued in BFS cavity search".
        // self.current_visit_id + 1 represents "confirmed inside the cavity".
        self.current_visit_id += 2;
        if self.current_visit_id >= u32::MAX - 2 {
            self.visited.fill(0);
            self.current_visit_id = 1;
        }

        let mut cavity = Vec::new();
        let mut queue = vec![t_containing];
        self.visited[t_containing] = self.current_visit_id;

        let mut q_head = 0;
        while q_head < queue.len() {
            let t_idx = queue[q_head];
            q_head += 1;

            if self.in_circumsphere(t_idx, p) {
                cavity.push(t_idx);
                self.visited[t_idx] = self.current_visit_id + 1; // Mark as inside cavity
                for &neighbor in &self.tetrahedra[t_idx].neighbors {
                    if let Some(n_idx) = neighbor {
                        if self.visited[n_idx] < self.current_visit_id {
                            self.visited[n_idx] = self.current_visit_id;
                            queue.push(n_idx);
                        }
                    }
                }
            }
        }

        let mut boundary = Vec::new();
        for &t_idx in &cavity {
            let tet = &self.tetrahedra[t_idx];
            for face_idx in 0..4 {
                let neighbor = tet.neighbors[face_idx];
                let is_boundary = match neighbor {
                    None => true,
                    Some(n_idx) => self.visited[n_idx] != self.current_visit_id + 1,
                };
                if is_boundary {
                    boundary.push((t_idx, face_idx));
                }
            }
        }

        let new_tets_start_idx = self.tetrahedra.len();
        let mut new_tets = Vec::with_capacity(boundary.len());
        let mut edge_to_face = HashMap::with_capacity(boundary.len() * 2);

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
                let sorted_edge = if edge.0 < edge.1 { (edge.0, edge.1) } else { (edge.1, edge.0) };

                if let Some(&(other_tet_idx, other_face_idx)) = edge_to_face.get(&sorted_edge) {
                    new_tets[b_idx].neighbors[face_sub_idx] = Some(other_tet_idx);
                    if other_tet_idx < new_tets_start_idx {
                        self.tetrahedra[other_tet_idx].neighbors[other_face_idx] = Some(new_tet_idx);
                    } else {
                        new_tets[other_tet_idx - new_tets_start_idx].neighbors[other_face_idx] = Some(new_tet_idx);
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
            self.visited.push(0);
        }

        self.last_tet = new_tets_start_idx;
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
                (v[0], v[1]), (v[0], v[2]), (v[0], v[3]),
                (v[1], v[2]), (v[1], v[3]),
                (v[2], v[3])
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
                let t_containing = new_triangulation.locate_cell(new_triangulation.last_tet, p);
                new_triangulation.insert_at(i, p, t_containing);
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
    pub fn from_points(mut points: Vec<[f64; 3]>) -> Result<Self, Error> {
        sort_brio_spatial(&mut points);
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
    pub fn from_delaunay(mut points: Vec<[f64; 3]>) -> Result<Self, Error> {
        sort_brio_spatial(&mut points);
        let (_, neighbors_at_insertion) = Delaunay3dKernel::from_points(&points)?;
        Self::from_insertion_neighbors(points, neighbors_at_insertion)
    }
}
