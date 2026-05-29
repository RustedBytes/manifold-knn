use core::cmp::Ordering;

use crate::Neighbor;

#[inline]
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
    #[inline]
    pub(crate) const fn new_empty() -> Self {
        Self {
            capacity: 0,
            items: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn reset(&mut self, capacity: usize) {
        self.capacity = capacity;
        self.items.clear();
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[Neighbor] {
        &self.items
    }

    #[inline]
    pub(crate) fn insert(&mut self, neighbor: Neighbor) -> bool {
        if self.capacity == 0 {
            return false;
        }

        if self.items.len() == self.capacity
            && let Some(worst) = self.items.last()
            && neighbor_cmp(worst, &neighbor) != Ordering::Greater
        {
            return false;
        }

        if self.items.iter().any(|item| item.index == neighbor.index) {
            return false;
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
}
