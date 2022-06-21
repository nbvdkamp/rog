use std::{fmt, sync::Mutex};

pub struct Statistics {
    store: Mutex<StatisticsStore>,
}

#[derive(Clone, Copy)]
pub struct StatisticsStore {
    pub inner_node_traversals: u32,
    pub intersection_tests: u32,
    pub intersection_hits: u32,
    pub rays: u32,
    pub max_depth: usize,
    pub inner_nodes: u32,
    pub leaf_nodes: u32,
}

impl Statistics {
    pub fn new() -> Self {
        Statistics {
            store: Mutex::new(StatisticsStore {
                inner_node_traversals: 0,
                intersection_tests: 0,
                intersection_hits: 0,
                rays: 0,
                max_depth: 0,
                inner_nodes: 0,
                leaf_nodes: 0,
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

    pub fn count_max_depth(&self, _depth: usize) {
        #[cfg(feature = "stats")]
        {
            let mut store = self.store.lock().unwrap();
            store.max_depth = store.max_depth.max(_depth);
        }
    }

    pub fn count_inner_node(&self) {
        #[cfg(feature = "stats")]
        {
            self.store.lock().unwrap().inner_nodes += 1;
        }
    }

    pub fn count_leaf_node(&self) {
        #[cfg(feature = "stats")]
        {
            self.store.lock().unwrap().leaf_nodes += 1;
        }
    }

    pub fn format_header() -> String {
        format!(
            "{:<10} | {:<10} | {:<10} | {:<10} | {:<11} | {:<10}",
            "Nodes/ray", "Tests/ray", "Hits/test", "Max depth", "Inner nodes", "Leaf nodes"
        )
    }
}

impl fmt::Display for StatisticsStore {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{:<10.5} | {:<10.5} | {:<10.5} | {:<10} | {:<11} | {:<10}",
            self.inner_node_traversals as f32 / self.rays as f32,
            self.intersection_tests as f32 / self.rays as f32,
            self.intersection_hits as f32 / self.intersection_tests as f32,
            self.max_depth,
            self.inner_nodes,
            self.leaf_nodes,
        )
    }
}
