use std::{
    cell::Cell,
    ptr::{addr_of_mut, NonNull},
    time::{Duration, Instant},
};

#[derive(Clone)]
struct StatCounter {
    time_spent_generating_ir: Duration,
    time_spent_jitting: Duration,
    total_time_spent_evaluating: Duration,
    time_spent_parsing: Duration,
}

thread_local! {
    static ACTIVE_TIME_COUNTERS: Cell<u64> = Cell::new(0);
}

fn time_counter_activate(mask: u64) -> bool {
    ACTIVE_TIME_COUNTERS.with(|x| {
        let v = x.get();
        if v & mask != 0 {
            return false;
        } else {
            x.set(v | mask);
            return true;
        }
    })
}

fn time_counter_deactivate(not_mask: u64) {
    ACTIVE_TIME_COUNTERS.with(|x| x.set(x.get() & not_mask))
}

pub struct TimeCounter {
    slot: Option<NonNull<Duration>>,
    start: Instant,
    not_mask: u64,
}

impl TimeCounter {
    pub fn stop(self) {}
}

impl Drop for TimeCounter {
    fn drop(&mut self) {
        if let Some(slot) = self.slot {
            unsafe {
                *slot.as_ptr() += Instant::now() - self.start;
            }
            time_counter_deactivate(self.not_mask);
        }
    }
}

macro_rules! def_counter_constructor {
    ($name: ident, $field: ident, $index: literal) => {
        pub fn $name() -> TimeCounter {
            TimeCounter {
                slot: if time_counter_activate(1 << $index) {
                    unsafe { Some(NonNull::new_unchecked(addr_of_mut!(COUNTER.$field))) }
                } else {
                    None
                },
                start: Instant::now(),
                not_mask: !(1 << $index),
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

def_counter_constructor!(measure_ir_generation_time, time_spent_generating_ir, 0);
def_counter_constructor!(measure_jit_codegen_time, time_spent_jitting, 1);
def_counter_constructor!(
    measure_total_evaluation_time,
    total_time_spent_evaluating,
    2
);
def_counter_constructor!(measure_parsing_time, time_spent_parsing, 3);

pub fn print_stats() {
    let stats = unsafe { COUNTER.clone() };

    println!(
        "total time spent evaluating = {:.2}ms",
        stats.total_time_spent_evaluating.as_secs_f64() * 1000.
    );
    println!(
        "time spent generating IR = {:.2}ms",
        stats.time_spent_generating_ir.as_secs_f64() * 1000.
    );
    println!(
        "time spent generating JIT executables = {:.2}ms",
        stats.time_spent_jitting.as_secs_f64() * 1000.
    );
    println!(
        "time spent parsing Nix source = {:.2}ms",
        stats.time_spent_parsing.as_secs_f64() * 1000.
    );
}
