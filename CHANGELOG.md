# Changelog

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
