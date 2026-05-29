use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use manifold_knn::ManifoldKnn;

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
    for _ in 0..n {
        points.push([next(), next(), next()]);
    }
    points
}

fn bench_index_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("Index Build");

    // Benchmark complete successor table build (sequential vs parallel)
    for size in [500, 1000].iter() {
        let points = pseudo_random_points(*size);

        group.bench_with_input(
            BenchmarkId::new("from_complete_successors", size),
            size,
            |b, &_s| {
                b.iter(|| {
                    let _ = ManifoldKnn::<3>::from_complete_successors(points.clone()).unwrap();
                });
            },
        );
    }

    // Benchmark from insertion neighbors (sorting / validating lists)
    for size in [500, 1000].iter() {
        let points = pseudo_random_points(*size);
        let neighbors_at_insertion: Vec<Vec<usize>> = (0..*size)
            .map(|j| {
                // Mildly dense neighbor lists: every point connects to a few prior ones
                let mut list = Vec::new();
                for i in 0..j {
                    if (i + j) % 20 == 0 {
                        list.push(i);
                    }
                }
                list
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("from_insertion_neighbors", size),
            size,
            |b, &_s| {
                b.iter(|| {
                    let _ = ManifoldKnn::<3>::from_insertion_neighbors(
                        points.clone(),
                        neighbors_at_insertion.clone(),
                    )
                    .unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("Query");
    let size = 10000;
    let points = pseudo_random_points(size);
    // Sparse neighbor lists: every point connects to a few prior ones
    let neighbors_at_insertion: Vec<Vec<usize>> = (0..size)
        .map(|j| {
            let mut list = Vec::new();
            for i in j.saturating_sub(5)..j {
                list.push(i);
            }
            list
        })
        .collect();
    let index = ManifoldKnn::<3>::from_insertion_neighbors(points, neighbors_at_insertion).unwrap();
    let query = [0.5, 0.5, 0.5];
    let mut workspace = manifold_knn::QueryWorkspace::new();

    group.bench_function("knn_k_10", |b| {
        b.iter(|| {
            let _ = index
                .knn_with_workspace(&query, 10, &mut workspace)
                .unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, bench_index_build, bench_query);
criterion_main!(benches);
