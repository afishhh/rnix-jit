use std::{
    ptr::addr_of_mut,
    time::{Duration, Instant},
};

#[derive(Clone)]
struct StatCounter {
    time_spent_generating_ir: Duration,
    time_spent_jitting: Duration,
    total_time_spent_evaluating: Duration,
    time_spent_parsing: Duration,
}

pub struct TimeCounter {
    slot: *mut Duration,
    start: Instant,
}

impl TimeCounter {
    pub fn pause<T>(&mut self, fun: impl FnOnce() -> T) -> T {
        unsafe {
            *self.slot += Instant::now() - self.start;
        }
        let result = fun();
        self.start = Instant::now();
        result
    }
    pub fn stop(self) {}
}

impl Drop for TimeCounter {
    fn drop(&mut self) {
        unsafe {
            *self.slot += Instant::now() - self.start;
        }
    }
}

macro_rules! def_counter_constructor {
    ($name: ident, $field: ident) => {
        pub fn $name() -> TimeCounter {
            TimeCounter {
                slot: unsafe { addr_of_mut!(COUNTER.$field) },
                start: Instant::now(),
            }
        }
    };
}

impl StatCounter {
    const fn new() -> Self {
        Self {
            time_spent_generating_ir: Duration::ZERO,
            time_spent_jitting: Duration::ZERO,
            total_time_spent_evaluating: Duration::ZERO,
            time_spent_parsing: Duration::ZERO,
        }
    }
}

static mut COUNTER: StatCounter = StatCounter::new();

def_counter_constructor!(measure_ir_generation_time, time_spent_generating_ir);
def_counter_constructor!(measure_jit_codegen_time, time_spent_jitting);
def_counter_constructor!(measure_total_evaluation_time, total_time_spent_evaluating);
def_counter_constructor!(measure_parsing_time, time_spent_parsing);

pub fn print_stats() {
    let stats = unsafe { COUNTER.clone() };

    println!(
        "total time spent evaluating = {:.2}ms",
        stats.total_time_spent_evaluating.as_secs_f64() * 1000.
    );
    println!(
        "total time spent generating IR = {:.2}ms",
        stats.time_spent_generating_ir.as_secs_f64() * 1000.
    );
    println!(
        "total time spent generating JIT executables = {:.2}ms",
        stats.time_spent_jitting.as_secs_f64() * 1000.
    );
    println!(
        "total time spent parsing Nix source = {:.2}ms",
        stats.time_spent_parsing.as_secs_f64() * 1000.
    );
}
