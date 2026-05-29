use core::cmp::Ordering;

use crate::Neighbor;

pub(crate) fn neighbor_cmp(left: &Neighbor, right: &Neighbor) -> Ordering {
    left.squared_distance
        .total_cmp(&right.squared_distance)
        .then_with(|| left.index.cmp(&right.index))
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BoundedNeighbors {
    capacity: usize,
    items: Vec<Neighbor>,
}

impl BoundedNeighbors {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            items: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn insert(&mut self, neighbor: Neighbor) -> bool {
        if self.capacity == 0 {
            return false;
        }

        if self.items.iter().any(|item| item.index == neighbor.index) {
            return false;
        }

        if self.items.len() == self.capacity {
            if let Some(worst) = self.items.last() {
                if neighbor_cmp(worst, &neighbor) != Ordering::Greater {
                    return false;
                }
            }
        }

        let position = self
            .items
            .binary_search_by(|candidate| neighbor_cmp(candidate, &neighbor))
            .unwrap_or_else(|position| position);
        self.items.insert(position, neighbor);

        if self.items.len() > self.capacity {
            self.items.pop();
        }

        true
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &Neighbor> {
        self.items.iter()
    }

    pub(crate) fn into_vec(self) -> Vec<Neighbor> {
        self.items
    }
}
