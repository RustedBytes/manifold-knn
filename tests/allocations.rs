use manifold_knn::{ManifoldKnn, QueryWorkspace};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct TrackingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static IS_TRACKING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if IS_TRACKING.with(|t| t.get()) {
            ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: TrackingAllocator = TrackingAllocator;

#[test]
fn test_zero_allocations_during_queries() {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mantissa = state >> 12;
        (mantissa as f64) / ((1_u64 << 52) as f64)
    };

    let n = 100;
    let mut points = Vec::with_capacity(n);
    for _ in 0..n {
        points.push([next(), next(), next()]);
    }

    let index = ManifoldKnn::<3>::from_complete_successors(points).unwrap();
    let query = [0.5, 0.5, 0.5];

    let mut workspace = QueryWorkspace::new();

    // Warm up the workspace so its internal buffers are allocated to the required capacity
    let _ = index
        .knn_with_workspace(&query, 10, &mut workspace)
        .unwrap();

    // Now turn on tracking for the current thread
    IS_TRACKING.with(|t| t.set(true));

    // Record the allocation count before the test run
    let count_before = ALLOC_COUNT.load(Ordering::SeqCst);

    // Run multiple queries
    for _ in 0..100 {
        let results = index
            .knn_with_workspace(&query, 10, &mut workspace)
            .unwrap();
        assert_eq!(results.len(), 10);
    }

    // Check allocation count after the run
    let count_after = ALLOC_COUNT.load(Ordering::SeqCst);

    // Turn off tracking
    IS_TRACKING.with(|t| t.set(false));

    let diff = count_after - count_before;

    assert_eq!(diff, 0, "Query allocated {} times!", diff);
}
