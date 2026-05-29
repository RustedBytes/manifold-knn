use manifold_knn::{ManifoldKnn, SuccessorTable};

fn pseudo_random_points(n: usize) -> Vec<[f64; 3]> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mantissa = state >> 12;
        (mantissa as f64) / ((1_u64 << 52) as f64)
    };

    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        // Mildly manifold-like: points near a noisy spiral surface strip.
        let t = 8.0 * next() + (i as f64) * 0.013;
        let radius = 0.5 + next();
        let z = 0.25 * (2.0 * next() - 1.0);
        points.push([radius * t.cos(), radius * t.sin(), z]);
    }
    points
}

fn pseudo_random_queries(n: usize) -> Vec<[f64; 3]> {
    let mut state = 0x243f_6a88_85a3_08d3_u64;
    let mut next = || {
        state = state
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        let mantissa = state >> 12;
        (mantissa as f64) / ((1_u64 << 52) as f64)
    };

    (0..n)
        .map(|_| [3.0 * next() - 1.5, 3.0 * next() - 1.5, 3.0 * next() - 1.5])
        .collect()
}

#[test]
fn complete_successors_match_bruteforce_for_prefixes() {
    let index = ManifoldKnn::<3>::from_complete_successors(pseudo_random_points(64)).unwrap();
    let prefixes = [0, 1, 2, 7, 31, 64];
    let ks = [0, 1, 2, 3, 8, 80];

    for query in pseudo_random_queries(30) {
        for prefix in prefixes {
            for k in ks {
                let dp = index.knn_prefix(&query, k, prefix).unwrap();
                let brute = index.brute_force_knn_prefix(&query, k, prefix).unwrap();
                assert_eq!(dp, brute, "query={query:?}, prefix={prefix}, k={k}");
            }
        }
    }
}

#[test]
fn insertion_neighbors_can_build_complete_table() {
    let points = pseudo_random_points(32);
    let neighbors_at_insertion: Vec<Vec<usize>> =
        (0..points.len()).map(|j| (0..j).collect()).collect();
    let index = ManifoldKnn::<3>::from_insertion_neighbors(points, neighbors_at_insertion).unwrap();

    assert_eq!(index.successors(), &SuccessorTable::complete(32));

    for query in pseudo_random_queries(10) {
        assert_eq!(
            index.knn(&query, 6).unwrap(),
            index.brute_force_knn(&query, 6).unwrap()
        );
    }
}

#[test]
fn insertion_updates_successor_lists() {
    let mut index = ManifoldKnn::<3>::from_complete_successors(vec![[0.0, 0.0, 0.0]]).unwrap();
    let i1 = index.insert_with_neighbors([1.0, 0.0, 0.0], [0]).unwrap();
    let i2 = index
        .insert_with_neighbors([0.0, 1.0, 0.0], [0, i1])
        .unwrap();

    assert_eq!(i1, 1);
    assert_eq!(i2, 2);
    assert_eq!(index.successors().list(0), &[1, 2]);
    assert_eq!(index.successors().list(1), &[2]);

    let nn = index.knn(&[0.9, 0.1, 0.0], 2).unwrap();
    assert_eq!(nn[0].index, 1);
}

#[test]
fn delete_with_complete_table_keeps_exact_queries() {
    let mut index = ManifoldKnn::<3>::from_complete_successors(pseudo_random_points(40)).unwrap();
    index.delete_with_new_successors(7, []).unwrap();
    index.delete_with_new_successors(13, []).unwrap();

    for query in pseudo_random_queries(20) {
        for k in [1, 3, 10] {
            assert_eq!(
                index.knn(&query, k).unwrap(),
                index.brute_force_knn(&query, k).unwrap()
            );
        }
    }
}

#[test]
fn delete_rebuild_complete_is_exact_fallback() {
    let points = pseudo_random_points(25);
    let sparse_neighbors: Vec<Vec<usize>> = (0..points.len())
        .map(|j| if j == 0 { Vec::new() } else { vec![j - 1] })
        .collect();
    let mut index = ManifoldKnn::<3>::from_insertion_neighbors(points, sparse_neighbors).unwrap();
    index.delete_rebuild_complete(5).unwrap();

    for query in pseudo_random_queries(10) {
        assert_eq!(
            index.knn(&query, 5).unwrap(),
            index.brute_force_knn(&query, 5).unwrap()
        );
    }
}

#[test]
fn invalid_successor_tables_are_rejected() {
    let duplicate = SuccessorTable::try_from_lists(vec![vec![1, 1], vec![]]);
    assert!(duplicate.is_err());

    let backwards = SuccessorTable::try_from_lists(vec![vec![], vec![0]]);
    assert!(backwards.is_err());

    let normalized =
        SuccessorTable::from_lists_normalized(vec![vec![2, 1, 2], vec![2], vec![]]).unwrap();
    assert_eq!(normalized.list(0), &[1, 2]);
}

#[test]
#[cfg(feature = "parallel")]
fn parallel_features_produce_identical_results() {
    let points = pseudo_random_points(100);

    // Build index with parallel feature
    let index = ManifoldKnn::<3>::from_complete_successors(points.clone()).unwrap();

    // Manually build sequential expected table
    let mut expected_lists = Vec::with_capacity(points.len());
    for owner in 0..points.len() {
        let mut list = Vec::with_capacity(points.len().saturating_sub(owner + 1));
        for successor in (owner + 1)..points.len() {
            list.push(successor);
        }
        expected_lists.push(list);
    }
    let table_seq = SuccessorTable::try_from_lists(expected_lists).unwrap();
    assert_eq!(index.successors(), &table_seq);

    // Verify index queries yield same results
    for query in pseudo_random_queries(10) {
        let results = index.knn(&query, 5).unwrap();
        let brute = index.brute_force_knn(&query, 5).unwrap();
        assert_eq!(results, brute);
    }
}

#[test]
#[cfg(feature = "parallel")]
fn parallel_insertion_neighbors_produce_identical_results() {
    let points = pseudo_random_points(50);
    let neighbors_at_insertion: Vec<Vec<usize>> =
        (0..points.len()).map(|j| (0..j).collect()).collect();

    let index_parallel =
        ManifoldKnn::<3>::from_insertion_neighbors(points.clone(), neighbors_at_insertion.clone())
            .unwrap();

    assert_eq!(index_parallel.successors(), &SuccessorTable::complete(50));
}
