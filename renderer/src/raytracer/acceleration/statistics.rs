use std::{fmt, sync::Mutex};

#[derive(Default)]
pub struct Statistics {
    store: Mutex<StatisticsStore>,
}

#[derive(Clone, Copy, Default)]
pub struct StatisticsStore {
    pub inner_node_traversals: u32,
    pub intersection_tests: u32,
    pub intersection_hits: u32,
    pub rays: u32,
    pub max_depth: usize,
    pub inner_nodes: u32,
    pub leaf_nodes: u32,
}

use std::ops;

impl_op_ex!(+ |a: &StatisticsStore, b: &StatisticsStore| -> StatisticsStore {
    StatisticsStore {
        inner_node_traversals: a.inner_node_traversals + b.inner_node_traversals,
        intersection_tests: a.intersection_tests + b.intersection_tests,
        intersection_hits: a.intersection_hits + b.intersection_hits,
        rays: a.rays + b.rays,
        max_depth: a.max_depth.max(b.max_depth),
        inner_nodes: a.inner_nodes + b.inner_nodes,
        leaf_nodes: a.leaf_nodes + b.leaf_nodes,
    }
});

impl Statistics {
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

    #[cfg(feature = "stats")]
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
