use std::sync::Mutex;

pub struct Statistics {
    store: Mutex<StatisticsStore>,
}

#[derive(Clone, Copy)]
pub struct StatisticsStore {
    pub inner_node_traversals: u32,
    pub intersection_tests: u32,
    pub intersection_hits: u32,
    pub rays: u32,
}

impl Statistics {
    pub fn new() -> Self {
        Statistics {
            store: Mutex::new(StatisticsStore {
                inner_node_traversals: 0,
                intersection_tests: 0,
                intersection_hits: 0,
                rays: 0,
            }),
        }
    }

    pub fn get_copy(&self) -> StatisticsStore {
        self.store.lock().unwrap().to_owned()
    }

    pub fn count_inner_node_traversal(&self) {
        #[cfg(feature = "stats")]
        {
            self.store.lock().unwrap().inner_node_traversals += 1;
        }
    }

    pub fn count_intersection_test(&self) {
        #[cfg(feature = "stats")]
        {
            self.store.lock().unwrap().intersection_tests += 1;
        }
    }

    pub fn count_intersection_hit(&self) {
        #[cfg(feature = "stats")]
        {
            self.store.lock().unwrap().intersection_hits += 1;
        }
    }

    pub fn count_ray(&self) {
        #[cfg(feature = "stats")]
        {
            self.store.lock().unwrap().rays += 1;
        }
    }
}
