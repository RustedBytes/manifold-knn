#![cfg(feature = "delaunay-3d")]

use manifold_knn::{Delaunay3dKernel, DelaunayManifoldKnn3, ManifoldKnn, SuccessorTable};

fn points() -> Vec<[f64; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.25, 0.35, 0.90],
        [1.25, 0.20, 0.60],
    ]
}

fn queries() -> Vec<[f64; 3]> {
    vec![
        [0.10, 0.10, 0.10],
        [0.30, 0.30, 0.25],
        [0.90, 0.90, 0.95],
        [1.10, 0.25, 0.50],
    ]
}

#[test]
fn kernel_emits_valid_insertion_neighbors() -> Result<(), manifold_knn::Error> {
    let pts = points();
    let (kernel, neighbors_at_insertion) = Delaunay3dKernel::from_points(&pts)?;

    assert_eq!(kernel.len(), pts.len());
    assert_eq!(kernel.active_len(), pts.len());
    assert_eq!(neighbors_at_insertion.len(), pts.len());
    assert_eq!(neighbors_at_insertion[0], Vec::<usize>::new());

    for (inserted, neighbors) in neighbors_at_insertion.iter().enumerate() {
        assert!(neighbors.iter().all(|&neighbor| neighbor < inserted));
    }

    let table = SuccessorTable::from_insertion_neighbors(pts.len(), neighbors_at_insertion)?;
    table.validate_for_len(pts.len())?;
    Ok(())
}

#[test]
fn from_delaunay_matches_bruteforce_for_sample_queries() -> Result<(), manifold_knn::Error> {
    let index = ManifoldKnn::<3>::from_delaunay(points())?;

    for query in queries() {
        for k in [1, 2, 3, 10] {
            assert_eq!(
                index.knn(&query, k)?,
                index.brute_force_knn(&query, k)?,
                "query={query:?}, k={k}"
            );
        }
    }

    Ok(())
}

#[test]
fn synchronized_insert_matches_bruteforce() -> Result<(), manifold_knn::Error> {
    let mut index = DelaunayManifoldKnn3::from_points(points())?;
    let inserted = index.insert([0.30, 0.30, 0.20])?;

    assert_eq!(inserted, 7);
    assert!(index.is_active(inserted));

    for query in queries() {
        assert_eq!(
            index.knn(&query, 4)?,
            index.index().brute_force_knn(&query, 4)?
        );
    }

    Ok(())
}

#[test]
fn synchronized_delete_keeps_queries_exact_with_conservative_edges(
) -> Result<(), manifold_knn::Error> {
    let mut index = DelaunayManifoldKnn3::from_points(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])?;
    let inserted = index.insert([0.25, 0.25, 0.25])?;
    let report = index.delete(inserted)?;

    assert!(!index.is_active(inserted));
    assert!(report.removed_references > 0);

    for query in queries() {
        assert_eq!(
            index.knn(&query, 3)?,
            index.index().brute_force_knn(&query, 3)?
        );
    }

    Ok(())
}
