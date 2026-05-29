# manifold-knn

`manifold-knn` is a Rust implementation of the dynamic-programming k-nearest-neighbor query method from:

> Pengfei Wang, Qinghao Guo, Haisen Zhao, Shiqing Xin, Shuangmin Chen, Changhe Tu, Wenping Wang. **Manifold k-NN: Accelerated k-NN Queries for Manifold Point Clouds**. arXiv:2605.02224v1, 2026.

The crate keeps the query/data-structure layer small and explicit, and now includes an optional 3D Delaunay backend powered by [`acgetchell/delaunay`](https://github.com/acgetchell/delaunay).

## What is implemented

- Dynamic-programming 1-NN transition-site traversal.
- DP-based k-NN query over a successor table.
- Prefix-subset queries: `knn_prefix(query, k, prefix_len)` ignores candidates with birth index `>= prefix_len` without rebuilding.
- Successor-table construction from insertion-time Delaunay neighborhoods.
- Optional 3D Delaunay backend behind `--features delaunay-3d`:
  - `Delaunay3dKernel` inserts points into a real incremental 3D Delaunay triangulation.
  - `SuccessorTable::from_delaunay3d(points)` builds successor lists from 3D Delaunay insertions.
  - `ManifoldKnn::<3>::from_delaunay(points)` builds a query index directly.
  - `DelaunayManifoldKnn3` keeps the Delaunay backend and query index synchronized during insertions and deletions.
- Dynamic insertion hook: `insert_with_neighbors(point, prior_neighbors)`.
- Dynamic deletion hooks:
  - `delete_with_new_successors(index, edges)` for externally computed local Delaunay update edges.
  - `delete_rebuild_complete(index)` as an exact quadratic fallback for small data/tests.
- Exact brute-force helpers for validation.
- `#![deny(unsafe_code)]`.

## Cargo features

Default build:

```toml
[dependencies]
manifold-knn = { path = "manifold-knn-rs" }
```

3D Delaunay backend:

```toml
[dependencies]
manifold-knn = { path = "manifold-knn-rs", features = ["delaunay-3d"] }
```

The optional backend currently pins `delaunay = "=0.7.8"`. That crate declares Rust 1.95, so this package also declares Rust 1.95.

## Quick start

```rust
use manifold_knn::ManifoldKnn;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];

    let index = ManifoldKnn::<3>::from_complete_successors(points)?;
    let neighbors = index.knn(&[0.2, 0.1, 0.0], 2)?;

    for n in neighbors {
        println!("{} {}", n.index, n.squared_distance);
    }

    Ok(())
}
```

## Building directly from 3D Delaunay

Enable `delaunay-3d` and use the convenience constructor:

```rust
use manifold_knn::ManifoldKnn;

fn build() -> Result<ManifoldKnn<3>, manifold_knn::Error> {
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.2, 0.2, 0.2],
    ];

    ManifoldKnn::<3>::from_delaunay(points)
}
```

For dynamic point sets, use the synchronized wrapper:

```rust
use manifold_knn::DelaunayManifoldKnn3;

fn dynamic() -> Result<(), manifold_knn::Error> {
    let mut index = DelaunayManifoldKnn3::from_points(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])?;

    let inserted = index.insert([0.2, 0.2, 0.2])?;
    assert_eq!(inserted, 4);

    let neighbors = index.knn(&[0.25, 0.2, 0.2], 3)?;
    assert_eq!(neighbors.len(), 3);

    Ok(())
}
```

## Building from external Delaunay insertion neighborhoods

When point `j` is inserted, compute the earlier Delaunay neighbors of `j` in the prefix set. Pass those lists to `from_insertion_neighbors`:

```rust
use manifold_knn::ManifoldKnn;

fn build() -> Result<ManifoldKnn<3>, manifold_knn::Error> {
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ];

    // neighbors_at_insertion[j] contains i < j adjacent to j at insertion time.
    let neighbors_at_insertion = vec![
        vec![],       // p0
        vec![0],      // p1 was adjacent to p0 when inserted
        vec![0, 1],   // p2 was adjacent to p0 and p1 when inserted
    ];

    ManifoldKnn::<3>::from_insertion_neighbors(points, neighbors_at_insertion)
}
```

This constructor creates the successor table by appending `j` to every listed neighbor's successor list.

## Prefix queries

Birth order can represent a progressive scan, a coarse-to-fine ordering, or a temporal stream. Query a prefix without rebuilding:

```rust
use manifold_knn::ManifoldKnn;

fn query_prefix() -> Result<(), manifold_knn::Error> {
    let index = ManifoldKnn::<3>::from_complete_successors(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])?;

    let first_two_only = index.knn_prefix(&[1.8, 0.0, 0.0], 2, 2)?;
    assert!(first_two_only.iter().all(|n| n.index < 2));
    Ok(())
}
```

## Dynamic updates

Insertion is direct if your triangulation layer gives the prior neighbors of the new point:

```rust
use manifold_knn::ManifoldKnn;

fn insert() -> Result<(), manifold_knn::Error> {
    let mut index = ManifoldKnn::<3>::from_complete_successors(vec![[0.0, 0.0, 0.0]])?;
    let new_index = index.insert_with_neighbors([1.0, 0.0, 0.0], [0])?;
    assert_eq!(new_index, 1);
    Ok(())
}
```

For deletion, the exact fast paper method requires local retriangulation around the deleted site. Supply the new successor edges created by that local update:

```rust
use manifold_knn::ManifoldKnn;

fn delete() -> Result<(), manifold_knn::Error> {
    let mut index = ManifoldKnn::<3>::from_complete_successors(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])?;

    let report = index.delete_with_new_successors(1, [])?;
    println!("removed {} references", report.removed_references);
    Ok(())
}
```

`DelaunayManifoldKnn3::delete` removes the vertex in the upstream triangulation, runs flip-based Delaunay repair, and conservatively inserts all current active Delaunay graph edges into the successor table. This keeps queries exact, but it may make the table denser than the minimal local-cavity update after many deletions.

For small data, `delete_rebuild_complete` is a correctness-first fallback.

## Correctness notes

The DP k-NN query is exact when the successor table satisfies the paper's invariant: `j` appears in `successors[i]` exactly when insertion of `j` pruned the Voronoi cell of `i` in the prefix. The `from_complete_successors` table is also exact because it is a superset of all required successor edges, but it gives up the sparse-manifold acceleration.

Floating-point ties are ordered by birth index for deterministic output. Like the paper's geometric argument, best performance and cleanest behavior are expected under non-degenerate point sets.

The `delaunay-3d` backend inherits the upstream crate's behavior for degeneracy, insertion failures, and removal limitations. The wrapper reports those errors as `Error::DelaunayKernel` instead of silently falling back.

## Development

The intended checks are:

```bash
cargo fmt --all --check
cargo test
cargo test --features delaunay-3d
cargo clippy --all-targets --all-features -- -D warnings
```

The included tests compare DP queries against brute force under complete successor tables, prefix restrictions, insertion, deletion fallback, and feature-gated 3D Delaunay construction.
