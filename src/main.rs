#![allow(clippy::missing_transmute_annotations)]
#![allow(clippy::fn_to_numeric_cast)]
#![allow(clippy::type_complexity)]
// This is likely to be stalibised soon
#![feature(offset_of_nested)]
// This can be replaced later
#![feature(offset_of_enum)]
// Rust's comedic const eval support
#![feature(const_float_bits_conv)]

use std::{
    cell::UnsafeCell,
    ffi::CStr,
    fmt::{Debug, Write as FmtWrite},
    mem::offset_of,
    ops::Deref,
    path::PathBuf,
    process::ExitCode,
    rc::Rc,
};

use clap::{Parser, Subcommand};
use dwarf::*;

mod compiler;
mod ir;
mod perfstats;
mod value;
use ir::*;
use perfstats::{measure_parsing_time, measure_total_evaluation_time};
use value::*;

enum ScopeStorage {
    // Used for recursive attribute sets
    #[allow(dead_code)] // Used dynamically via JIT
    Attrset,
    Scope(ValueMap),
}

#[repr(C)]
struct Scope {
    values: *const ValueMap,
    previous: *mut Scope,
    storage: ScopeStorage,
}

impl Scope {
    fn from_map(map: ValueMap, previous: *mut Scope) -> *mut Scope {
        unsafe {
            let scope_ptr = Box::leak(Box::new(Scope {
                values: std::ptr::null(),
                storage: ScopeStorage::Scope(map),
                previous,
            })) as *mut Scope;
            (*scope_ptr).values =
                (scope_ptr as *const u8).add(offset_of!(Scope, storage.Scope.0)) as *const _;
            scope_ptr
        }
    }

    fn from_attrs(map: Rc<UnsafeCell<ValueMap>>, previous: *mut Scope) -> *mut Scope {
        Box::leak(Box::new(Scope {
            values: unsafe { (*Rc::<UnsafeCell<ValueMap>>::into_raw(map)).get() },
            previous,
            storage: ScopeStorage::Attrset,
        }))
    }

    unsafe fn lookup(mut scope: *mut Scope, name: &str) -> Value {
        loop {
            let mut scope_debug = String::new();
            UnpackedValue::fmt_attrset_display(&*(*scope).values, 0, &mut scope_debug, true)
                .unwrap();
            // eprintln!("{scope:?} {name} {}", scope_debug);
            match (*scope).get(name) {
                Some(value) => return value.clone(),
                None => {
                    scope = (*scope).previous;
                    if scope.is_null() {
                        throw!("identifier {name} not found")
                    }
                }
            }
        }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        if matches!(self.storage, ScopeStorage::Attrset) {
            unsafe { Rc::decrement_strong_count(self.values) }
        }
    }
}

impl Deref for Scope {
    type Target = ValueMap;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.values }
    }
}

mod dwarf;

mod exception;
mod unwind;
use exception::*;

mod builtins;
use builtins::{import, seq, ROOT_SCOPE};

#[derive(Parser, Clone, Debug)]
struct Opts {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Clone, Debug)]
enum Command {
    Repl,
    Eval(EvalCommand),
}

#[derive(Parser, Clone, Debug)]
// enum groups are not currently implemented in clap
// (issue: https://github.com/clap-rs/clap/issues/2621)
#[group(multiple = false)]
struct EvalCommand {
    #[clap(short, long)]
    file: Option<PathBuf>,
    #[clap(short, long)]
    expr: Option<String>,
}

fn pretty_print_exception(exception: NixException) {
    eprintln!("\x1b[31;1merror:\x1b[0m {}", exception.message);
    eprintln!("backtrace (most recent first):");
    for (i, span) in exception.stacktrace.into_iter().enumerate() {
        let source = &span.file.content;

        let span_line_start = source[..span.range.start().into()]
            .rfind('\n')
            .map(|x| x + 1)
            .unwrap_or(0);
        let span_line_end = source[span.range.end().into()..]
            .find('\n')
            .map(|x| x + Into::<usize>::into(span.range.end()))
            .unwrap_or(source.len());

        let nlines = source[span_line_start..span_line_end]
            .chars()
            .filter(|c| *c == '\n')
            .count()
            + 1;
        let lineno = source[..span.range.start().into()]
            .chars()
            .filter(|x| *x == '\n')
            .count()
            + 1;
        let colno = Into::<usize>::into(span.range.start()) - span_line_start + 1;
        let final_line_index = source[..span.range.end().into()]
            .rfind('\n')
            .map(|x| x + 1)
            .unwrap_or(0);
        let end_colno = Into::<usize>::into(span.range.end()) - final_line_index;
        let end_offset = if final_line_index == span_line_start {
            end_colno - colno + 1
        } else {
            end_colno
        };

        eprintln!("\t{i}: {}:{}:{}", span.file.filename, lineno, colno);
        let mut is_highlighted = false;
        for (i, mut line) in source[span_line_start..span_line_end].lines().enumerate() {
            eprint!("\t{:>6}|", i + lineno);
            if i == 0 {
                eprint!("{}\x1b[33m", &line[..colno - 1]);
                line = &line[colno - 1..];
                is_highlighted = true;
            }

            if is_highlighted {
                eprint!("\x1b[33m");
            }

            if i == nlines - 1 {
                eprint!("{}\x1b[0m", &line[..end_offset]);
                line = &line[end_offset..];
                is_highlighted = false;
            }

            eprintln!("{}\x1b[0m", line);
        }
        eprintln!()
    }
}

fn eval(root: PathBuf, filename: String, code: String) -> Result<Value, NixException> {
    let parse = {
        let _tc = measure_parsing_time();
        rnix::Root::parse(&code)
    };
    if !parse.errors().is_empty() {
        let mut msg = "parse errors encountered:\n".to_string();
        for (i, error) in parse.errors().iter().enumerate() {
            writeln!(msg, "{}. {}", i + 1, error).unwrap();
        }
        return Err(NixException::new(msg));
    }
    let expr = parse.tree().expr().unwrap();
    let program = IRCompiler::compile(root, filename, code, expr);
    let executable = unsafe { compiler::COMPILER.compile(program, None).unwrap() };
    catch_nix_unwind(move || ROOT_SCOPE.with(move |s| executable.run(*s, &Value::NULL)))
}

fn main() -> ExitCode {
    let opts = Opts::parse();
    match opts.command {
        Command::Repl => loop {
            let line = unsafe {
                let line_ptr = readline_sys::readline(c"> ".as_ptr());
                if line_ptr.is_null() {
                    return ExitCode::SUCCESS;
                }
                let line = CStr::from_ptr(line_ptr);
                let string = line.to_str().unwrap().to_string();
                if !line.is_empty() {
                    readline_sys::add_history(line_ptr);
                }
                nix::libc::free(line.as_ptr() as *mut std::ffi::c_void);
                string
            };
            match eval(std::env::current_dir().unwrap(), "<repl>".to_string(), line).and_then(
                |value| {
                    catch_nix_unwind(|| {
                        seq(&value, true);
                        value
                    })
                },
            ) {
                Ok(value) => println!("{value}"),
                Err(exception) => pretty_print_exception(exception),
            }
        },
        Command::Eval(EvalCommand { file, expr }) => {
            match catch_nix_unwind(move || {
                let _tc = measure_total_evaluation_time();
                let value = if let Some(file) = file {
                    import(file)
                } else if let Some(expr) = expr {
                    eval(
                        std::env::current_dir().unwrap(),
                        "<string>".to_string(),
                        expr,
                    )?
                } else {
                    unreachable!()
                };
                seq(&value, true);
                Ok(value)
            })
            // me when Result::flatten is unstable
            .and_then(|x| x)
            {
                Ok(value) => {
                    println!("{value:?}");
                    perfstats::print_stats();
                    ExitCode::SUCCESS
                }
                Err(exception) => {
                    pretty_print_exception(exception);
                    perfstats::print_stats();
                    ExitCode::FAILURE
                }
            }
        }
    }
}
