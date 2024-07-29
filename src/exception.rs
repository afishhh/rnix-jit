pub const NIX_EXCEPTION_CLASS_SLICE: &[u8; 8] = b"FSH\0NIX\0";
pub const NIX_EXCEPTION_CLASS: u64 = u64::from_le_bytes(*NIX_EXCEPTION_CLASS_SLICE);

use std::{arch::global_asm, fmt::Debug, mem::ManuallyDrop, sync::OnceLock};

use crate::{dwarf::*, unwind::*, SourceSpan};

#[repr(C)]
pub struct NixException {
    base: _Unwind_Exception,
    pub message: String,
    pub stacktrace: Vec<SourceSpan>,
}

impl NixException {
    pub fn new(message: String) -> Box<Self> {
        Box::new(Self {
            base: _Unwind_Exception::new(NIX_EXCEPTION_CLASS, Self::cleanup),
            message,
            stacktrace: vec![],
        })
    }

    pub fn raise(self: Box<Self>) -> ! {
        let result =
            unsafe { _Unwind_RaiseException(&mut Box::leak(self).base as *mut _Unwind_Exception) };
        println!("NixException::raise: _Unwind_RaiseException failed: {result:?}");
        std::process::abort();
    }

    unsafe extern "C" fn cleanup(code: _Unwind_Reason_Code, object: *mut _Unwind_Exception) {
        let this = object as *mut NixException;
        drop(Box::from_raw(object as *mut NixException));
    }
}

impl Debug for NixException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NixException")
            .field("message", &self.message)
            .field("stacktrace", &self.stacktrace)
            .finish()
    }
}

global_asm!(
    r"
.global rnix_jit_catch_nix_unwind
.global rnix_jit_catch_nix_unwind_ret
.global rnix_jit_catch_nix_unwind_end
.global rnix_jit_catch_nix_unwind_return_tagged_err
rnix_jit_catch_nix_unwind:
    push rdx
    call rsi
    add rsp, 8
rnix_jit_catch_nix_unwind_ret:
    ret
rnix_jit_catch_nix_unwind_end:
rnix_jit_catch_nix_unwind_return_tagged_err:
    mov rax, rdx # ptr
    mov rdx, 1 # is_error = true
    add rsp, 8
    ret
"
);

extern "C" {
    fn rnix_jit_catch_nix_unwind(
        fun_arg: u64,
        fun: unsafe extern "C-unwind" fn(arg: u64) -> PtrResult,
    ) -> PtrResult;
    static rnix_jit_catch_nix_unwind_ret: std::ffi::c_void;
    static rnix_jit_catch_nix_unwind_end: std::ffi::c_void;
    // fake signature: error pointer is actually passed in rdx
    fn rnix_jit_catch_nix_unwind_return_tagged_err() -> PtrResult;
}

#[repr(C)]
#[derive(Debug)]
struct PtrResult {
    ptr: u64,
    is_error: bool,
}

unsafe extern "C" fn catch_nix_unwind_personality(
    version: std::ffi::c_int,
    actions: std::ffi::c_int,
    exception_class: u64,
    exception: *const _Unwind_Exception,
    context: *const _Unwind_Context,
) -> _Unwind_Reason_Code {
    if actions & _UA_FORCE_UNWIND > 0 {
        _URC_CONTINUE_UNWIND
    } else if actions & _UA_SEARCH_PHASE > 0 {
        if exception_class == NIX_EXCEPTION_CLASS {
            _URC_HANDLER_FOUND
        } else {
            _URC_CONTINUE_UNWIND
        }
    } else if actions & _UA_CLEANUP_PHASE > 0 {
        if actions & _UA_HANDLER_FRAME > 0 {
            let context = context as *mut _;
            _Unwind_SetIP(context, rnix_jit_catch_nix_unwind_return_tagged_err as u64);
            // FIXME: HACK: It would make sense to use rdi here... but that causes a segmentation fault
            //              I have no idea what preconditions are required for a register to be
            //              writable here, but inspecting the context structure in gdb shows that rdx
            //              is so just use that... (hacky)
            _Unwind_SetGR(context, /* rdx */ 1, exception as u64);
            _URC_INSTALL_CONTEXT
        } else {
            // no cleanup needed
            _URC_CONTINUE_UNWIND
        }
    } else {
        panic!("catch_nix_unwind_personality called with unexpected actions 0x{actions:x}")
    }
}

struct RegisteredFrame {
    data: Box<[u8]>,
}

impl Drop for RegisteredFrame {
    fn drop(&mut self) {
        unsafe { __deregister_frame(self.data.as_ptr()) }
    }
}

fn rnix_jit_catch_nix_unwind_init() -> RegisteredFrame {
    let eh_frame = Vec::into_boxed_slice(
        EhFrameBuilder::new(1, -8, catch_nix_unwind_personality, |_cie| {})
            .add_fde(
                rnix_jit_catch_nix_unwind as u64,
                unsafe {
                    &rnix_jit_catch_nix_unwind_end as *const _ as u64
                        - rnix_jit_catch_nix_unwind as u64
                },
                |out| {
                    DW_CFA_def_cfa(out, 7, 16);
                    DW_CFA_offset(out, 16, 1);
                    DW_CFA_advance_loc(out, unsafe {
                        &rnix_jit_catch_nix_unwind_ret as *const _ as u64
                            - rnix_jit_catch_nix_unwind as u64
                    } as u8);
                    DW_CFA_def_cfa(out, 7, 8);
                },
            )
            .build(),
    );
    unsafe { __register_frame(eh_frame.as_ptr()) };
    RegisteredFrame { data: eh_frame }
}

static INIT_CATCH_NIX_UNWIND: OnceLock<RegisteredFrame> = OnceLock::new();

// TODO: This function makes the return value go through the heap
//       It should not be called often but maybe there's a better way?
// SAFETY: Pretty unsafe
pub fn catch_nix_unwind<T, F: FnOnce() -> T>(function: F) -> Result<T, NixException> {
    INIT_CATCH_NIX_UNWIND.get_or_init(rnix_jit_catch_nix_unwind_init);

    // dropped inside the rnix_jit_catch_nix_unwind callback
    let mut function = ManuallyDrop::new(function);
    let erased_function = &mut function as *mut ManuallyDrop<F> as u64;

    unsafe extern "C-unwind" fn call_callback<T, F: FnOnce() -> T>(erased: u64) -> PtrResult {
        PtrResult {
            ptr: Box::into_raw(Box::new(ManuallyDrop::into_inner(
                (erased as *mut ManuallyDrop<F>).read(),
            )())) as u64,
            is_error: false,
        }
    }

    unsafe {
        let PtrResult { ptr, is_error } = rnix_jit_catch_nix_unwind(
            erased_function,
            call_callback::<T, F> as unsafe extern "C-unwind" fn(arg: u64) -> PtrResult,
        );
        if is_error {
            Err(*Box::from_raw(ptr as *mut NixException))
        } else {
            Ok(*Box::from_raw(ptr as *mut T))
        }
    }
}
