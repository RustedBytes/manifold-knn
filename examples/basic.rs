use manifold_knn::ManifoldKnn;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];

    let index = ManifoldKnn::<3>::from_complete_successors(points)?;
    let neighbors = index.knn(&[0.2, 0.1, 0.0], 3)?;

    for neighbor in neighbors {
        println!(
            "index={} squared_distance={:.6}",
            neighbor.index, neighbor.squared_distance
        );
    }

    Ok(())
}
