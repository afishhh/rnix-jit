use std::{ffi::CStr, path::PathBuf, process::ExitCode};

use clap::{Parser, Subcommand};

use rnix_jit::*;

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
    #[clap(long)]
    expr: Option<String>,
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
                let _tc = perfstats::measure_total_evaluation_time();
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
