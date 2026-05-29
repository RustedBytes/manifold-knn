# Changelog

## 0.7.3

- Optimized successor table layout using a flat Compressed Sparse Row (CSR) array, reducing heap allocations from one allocation per point to exactly two flat vectors, improving traversal cache locality and construction speeds.
- Pre-counted and slice-constructed successor arrays, eliminating intermediate dynamic allocations during index construction.
- Introduced `discovered` tracker to `QueryWorkspace` to prevent duplicate distance calculations and redundant candidate list insertions, improving query speeds.
- Added cross-crate inlining annotations on query-critical functions.
- Expanded the benchmark suite to include a `knn_k_100` case.

## 0.7.1

- Implemented a walk-based point location algorithm starting from the last inserted tetrahedron, combined with Biased Randomized Incremental Ordering (BRIO) and spatial Z-order sorting to achieve average-case $O(1)$ point location complexity without needing any KD-Tree.

## 0.7.0

- Replaced external generic `delaunay` crate dependency with a custom, highly optimized 3D Delaunay triangulation.
- Integrated `robust` crate (Jonathan Shewchuk's adaptive precision predicates) to achieve allocation-free orientation and in-sphere queries.
- Added an incremental 3D KD-Tree to accelerate point location queries from $O(N^{4/3})$ to $O(N \log N)$ by starting the visibility walk from the nearest already-inserted vertex.
- Protected against numerical degeneracies caused by coincident/duplicate point insertions.
- Achieved a ~40% to 61% speedup in 3D Delaunay index construction.

## 0.6.2

- Add SIMD optimizations

## 0.6.1

- Fixes

## 0.6.0

- Optimized query execution path by avoiding $O(N)$ linear scans over active point status and $O(N)$ workspace processed resets per query.
- Replaced global workspace resets with $O(k)$ localized resets tracking only visited indices.
- Achieved a ~370x (37,000%) query speedup for sparse/manifold point cloud query workloads.
- Maintained zero memory allocations at query time and full backward compatibility.

## 0.5.0

- Added optional `parallel` feature backed by `rayon = "1.10"` to speed up index building, sorting, and validation.
- Added a Criterion benchmark suite to compare sequential and parallel index building performance.

## 0.4.0

- Added `QueryWorkspace` struct for reusable buffers during nearest-neighbor queries.
- Added workspace-based query methods (`knn_with_workspace`, `knn_prefix_with_workspace`, `nearest_with_workspace`, etc.) enabling 100% allocation-free queries.
- Modified standard query methods (`knn`, `knn_prefix`, `nearest`, etc.) to reuse thread-local query workspaces, reducing internal allocations.
- Optimized candidate filtering in BoundedNeighbors by checking against the worst neighbor before performing duplicate scans.
- Added `#[inline]` annotations to hot functions like `squared_distance` and helper accessors.
- Added a tracking allocator integration test to verify zero-allocation behavior.

## 0.3.0

- Added optional `delaunay-3d` feature backed by `delaunay = "=0.7.8"`.
- Added `Delaunay3dKernel` for birth-order incremental 3D Delaunay insertions and incident-edge neighbor extraction.
- Added `DelaunayTriangulation3` type alias for the concrete upstream 3D triangulation type.
- Added `SuccessorTable::from_delaunay3d(points)`.
- Added `ManifoldKnn::<3>::from_delaunay(points)`.
- Added `DelaunayManifoldKnn3`, a synchronized Delaunay/query wrapper for dynamic 3D inserts and conservative dynamic deletes.
- Added Delaunay backend documentation, feature-gated tests, and a feature-gated example.
- Raised declared Rust version to 1.95 to match the optional upstream Delaunay dependency.

## 0.1.0

- Initial implementation of the Manifold k-NN query algorithm over successor tables.
- Prefix queries, insertion hooks, deletion hooks, complete-table fallback, and brute-force validation helpers.
