# Changelog

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
