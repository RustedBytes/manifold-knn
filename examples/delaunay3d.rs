#[cfg(feature = "delaunay-3d")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use manifold_knn::DelaunayManifoldKnn3;

    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.25, 0.35, 0.90],
    ];

    let mut index = DelaunayManifoldKnn3::from_points(points)?;
    let inserted = index.insert([0.30, 0.30, 0.20])?;
    println!("inserted birth index {inserted}");

    for neighbor in index.knn(&[0.28, 0.30, 0.22], 3)? {
        println!(
            "index={} distance={:.6}",
            neighbor.index,
            neighbor.distance()
        );
    }

    Ok(())
}

#[cfg(not(feature = "delaunay-3d"))]
fn main() {
    eprintln!("run with: cargo run --example delaunay3d --features delaunay-3d");
}
