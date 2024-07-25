//! Raw bindings to the System V unwinding API.
#![allow(non_camel_case_types)]

use std::ffi::{c_int, c_void};

pub type _Unwind_Exception_Cleanup_Fn =
    unsafe extern "C" fn(_Unwind_Reason_Code, *const _Unwind_Exception);
pub type _Unwind_Personality_Fn = unsafe extern "C" fn(
    c_int,
    _Unwind_Action,
    _Unwind_Exception_Class,
    *const _Unwind_Exception,
    *const _Unwind_Context,
) -> _Unwind_Reason_Code;

pub type _Unwind_Exception_Class = u64;
pub type _Unwind_Action = c_int;

pub type _Unwind_Stop_Fn = extern "C" fn(
    c_int,
    _Unwind_Action,
    _Unwind_Exception_Class,
    *const _Unwind_Exception,
    *const _Unwind_Context,
    *const c_void,
) -> _Unwind_Reason_Code;

#[repr(C)]
pub struct _Unwind_Exception {
    pub class: _Unwind_Exception_Class,
    pub cleanup: _Unwind_Exception_Cleanup_Fn,
    private_1: u64,
    private_2: u64,
}

impl _Unwind_Exception {
    pub fn new(class: _Unwind_Exception_Class, cleanup: _Unwind_Exception_Cleanup_Fn) -> Self {
        Self {
            class,
            cleanup,
            private_1: 0,
            private_2: 0
        }
    }
}

#[repr(C)]
pub struct _Unwind_Context {
    _inner: (),
}

extern "C" {
    pub fn __register_frame(pointer: *const u8);
    pub fn _Unwind_RaiseException(exception: *const _Unwind_Exception) -> _Unwind_Reason_Code;
    pub fn _Unwind_Resume(exception: *const _Unwind_Exception);
    pub fn _Unwind_DeleteException(exception: *mut _Unwind_Exception);
    pub fn _Unwind_GetGR(context: *const _Unwind_Context, index: c_int) -> u64;
    pub fn _Unwind_SetGR(context: *mut _Unwind_Context, index: c_int, newvalue: u64);
    pub fn _Unwind_GetIP(context: *const _Unwind_Context) -> u64;
    pub fn _Unwind_SetIP(context: *mut _Unwind_Context, newvalue: u64);
    pub fn _Unwind_GetLanguageSpecificData(context: *const _Unwind_Context) -> u64;
    pub fn _Unwind_GetRegionStart(context: *const _Unwind_Context) -> u64;
    pub fn _Unwind_ForcedUnwind(
        exception: *const _Unwind_Exception,
        stop_fn: _Unwind_Stop_Fn,
        data: *const c_void,
    ) -> _Unwind_Reason_Code;
    pub fn _Unwind_GetCFA(context: *const _Unwind_Context) -> u64;
    // libgcc also contains these but they don't seem to be defined by the ABI
    // alias (_Unwind_Backtrace);
    // alias (_Unwind_FindEnclosingFunction);
    // alias (_Unwind_GetDataRelBase);
    // alias (_Unwind_GetTextRelBase);
    // alias (_Unwind_Resume_or_Rethrow);
}

#[repr(C)]
#[derive(Debug)]
pub enum _Unwind_Reason_Code {
    _URC_NO_REASON = 0,
    _URC_FOREIGN_EXCEPTION_CAUGHT = 1,
    _URC_FATAIL_PHASE2_ERROR = 2,
    _URC_FATAIL_PHASE1_ERROR = 3,
    _URC_NORMAL_STOP = 4,
    _URC_END_OF_STACK = 5,
    _URC_HANDLER_FOUND = 6,
    _URC_INSTALL_CONTEXT = 7,
    _URC_CONTINUE_UNWIND = 8,
}

pub use _Unwind_Reason_Code::*;

pub const _UA_SEARCH_PHASE: c_int = 1;
pub const _UA_CLEANUP_PHASE: c_int = 2;
pub const _UA_HANDLER_FRAME: c_int = 4;
pub const _UA_FORCE_UNWIND: c_int = 8;
