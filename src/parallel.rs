use std::sync::Once;

static INIT: Once = Once::new();

/// Configure Rayon once from RSLIBHNM_THREADS; otherwise use Rayon default.
/// Invalid, zero, or oversized values are ignored. This is portable and does
/// not require BLAS/OpenMP runtime control.
pub fn configure_threads() {
    INIT.call_once(|| {
        let threads = std::env::var("RSLIBHNM_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0);
        if let Some(threads) = threads {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global();
        }
    });
}
