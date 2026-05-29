# Algorithm notes

This crate models the data structure from *Manifold k-NN: Accelerated k-NN Queries for Manifold Point Clouds* as two layers.

## Successor table

For a birth-ordered point set `p_0, p_1, ...`, a successor edge `i -> j` with `i < j` means insertion of `p_j` pruned the Voronoi cell of `p_i` in the prefix set. In an incremental Delaunay implementation, when `p_j` is inserted, append `j` to every earlier Delaunay neighbor of `j`.

The query algorithm assumes each successor list is sorted by birth index. `SuccessorTable` validates this invariant.

## 1-NN transition traversal

A query starts at the first active point in the prefix. It scans that point's successor list in insertion order. If it finds a successor that is closer to the query than the current point, that successor becomes the current point, and the traversal jumps to the new point's successor list. The visited current points are transition sites: each was nearest to the query at some prefix time.

## k-NN expansion

The k-NN query maintains a bounded sorted candidate list. It is initialized with transition sites from the 1-NN traversal. Then it repeatedly expands the successor list of the best unprocessed candidate. Every active successor inside the requested prefix is inserted into the bounded list if it ranks among the current best `k` candidates.

With a valid sparse successor table, this follows the theorem in the paper: every subsequent nearest neighbor is either a transition-site prefix candidate or lies in the successor list of an already discovered closer neighbor.

## Optional 3D Delaunay backend

With the `delaunay-3d` Cargo feature, the crate can build the successor table from a real 3D incremental Delaunay triangulation:

1. `Delaunay3dKernel` inserts points in birth order into `delaunay::DelaunayTriangulation`.
2. The wrapper stores the birth index as vertex data and maintains a stable map from upstream `VertexKey` values to local point indices.
3. After inserting a vertex, it reads `incident_edges(new_key)`, maps the edge endpoints back to birth indices, and records only earlier active neighbors.
4. Those insertion-neighbor lists are converted to the Manifold k-NN successor table by appending `j` to every earlier neighbor's successor list.

The first `D + 1` vertices are the bootstrap prefix for a 3D triangulation. Before the initial tetrahedron exists, the wrapper conservatively reports all earlier points as predecessors. This keeps the successor table reachable and exact for very small prefixes.

## Dynamic deletion with the backend

The paper's fast deletion update needs the Delaunay edges created by local cavity retriangulation after the deleted site is removed. The upstream `delaunay` crate exposes vertex removal, but it does not expose a minimal “new local successor edges” delta in the public API used here. Therefore `DelaunayManifoldKnn3::delete` is conservative:

1. Remove the vertex from the upstream triangulation.
2. Run flip-based Delaunay repair on the backend.
3. Enumerate all current active Delaunay graph edges.
4. Insert those edges into the successor table using `delete_with_new_successors`.

This is exact because it adds a superset of the local update edges. It can become denser than the minimal paper data structure after many deletions, so performance-sensitive applications can still call `delete_with_new_successors` directly with a custom local-cavity edge set.

## Complete table fallback

`SuccessorTable::complete(n)` stores all edges `i -> j` for `i < j`. This is a superset of every possible valid successor table, so queries are exact for arbitrary point sets and active masks. It is quadratic and intended for tests, small data, and deletion fallback.
