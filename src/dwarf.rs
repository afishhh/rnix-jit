#![allow(non_snake_case)]
#![allow(dead_code)]

// TODO: Add tests (have to get examples from somewhere...)

pub fn uleb128(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut current = (value & 0x7F) as u8;
        value >>= 7;
        if value > 0 {
            current |= 1 << 7;
        }
        out.push(current);
        if value == 0 {
            break;
        }
    }
}

// Algorithm from DWARF specification Figure C.2 "Algorithm to encode a signed integer"
pub fn sleb128(out: &mut Vec<u8>, mut value: i64) {
    let mut more = true;
    let negative = value < 0;
    let size = i64::BITS;
    while more {
        let mut byte = (value & 0x7F) as i8;
        value >>= 7;
        if negative {
            value |= -(1 << (size - 7));
        }
        if (value == 0 && (byte & 0x40) == 0) || (value == -1 && (byte & 0x40) != 0) {
            more = false;
        } else {
            byte |= 1 << 7;
        }
        out.push(unsafe { std::mem::transmute_copy::<i8, u8>(&byte) })
    }
}

macro_rules! define_dwarf_op {
    ($name: ident, $code: literal $(, $operand_name: ident: $operand_type: tt)*) => {
        pub fn $name(out: &mut Vec<u8> $(, $operand_name: define_dwarf_op!(@operand_param_type $operand_type))*) {
            out.push($code);
            $(define_dwarf_op!(@push_operand $operand_type, out, $operand_name);)*
        }
    };
    (@operand_param_type 1u) => { u8 };
    (@operand_param_type 1s) => { i8 };
    (@operand_param_type 2u) => { u16 };
    (@operand_param_type 2s) => { i16 };
    (@operand_param_type 4u) => { u32 };
    (@operand_param_type 4s) => { i32 };
    (@operand_param_type 8u) => { u64 };
    (@operand_param_type 8s) => { i64 };
    (@operand_param_type uleb128) => { u64 };
    (@operand_param_type sleb128) => { i64 };
    (@operand_param_type address) => { usize };
    (@push_operand 1u, $out: ident, $value: ident) => { $out.extend($value.to_le_bytes()); };
    (@push_operand 1s, $out: ident, $value: ident) => { $out.extend($value.to_le_bytes()); };
    (@push_operand 2u, $out: ident, $value: ident) => { $out.extend($value.to_le_bytes()); };
    (@push_operand 2s, $out: ident, $value: ident) => { $out.extend($value.to_le_bytes()); };
    (@push_operand 4u, $out: ident, $value: ident) => { $out.extend($value.to_le_bytes()); };
    (@push_operand 4s, $out: ident, $value: ident) => { $out.extend($value.to_le_bytes()); };
    (@push_operand 8u, $out: ident, $value: ident) => { $out.extend($value.to_le_bytes()); };
    (@push_operand 8s, $out: ident, $value: ident) => { $out.extend($value.to_le_bytes()); };
    (@push_operand address, $out: ident, $value: ident) => { $out.extend($value.to_le_bytes()); };
    (@push_operand uleb128, $out: ident, $value: ident) => { uleb128($out, $value); };
    (@push_operand sleb128, $out: ident, $value: ident) => { sleb128($out, $value); };
}

macro_rules! define_dwarf_ops {
    {
        $($name: ident[$opcode: literal](
            $($aname: tt: $atype: tt),*
        );)*
    } => {
        $(define_dwarf_op!($name, $opcode $(, $aname: $atype)*);)*
    };
}

// I left out some operations that took non-zero effort to implement and I knew would never get
// used in this project internally.
define_dwarf_ops! {
    DW_OP_deref[0x06]();
    DW_OP_const1u[0x08](value: 1u);
    DW_OP_const1s[0x09](value: 1s);
    DW_OP_const2u[0x0a](value: 2u);
    DW_OP_const2s[0x0b](value: 2s);
    DW_OP_const4u[0x0c](value: 4u);
    DW_OP_const4s[0x0d](value: 4s);
    DW_OP_const8u[0x0e](value: 8u);
    DW_OP_const8s[0x0f](value: 8s);
    DW_OP_constu[0x10](value: uleb128);
    DW_OP_consts[0x11](value: sleb128);
    DW_OP_dup[0x12]();
    DW_OP_drop[0x13]();
    DW_OP_over[0x14]();
    DW_OP_pick[0x15](value: 1u);
    DW_OP_swap[0x16]();
    DW_OP_rot[0x17]();
    DW_OP_xderef[0x18]();
    DW_OP_abs[0x19]();
    DW_OP_and[0x1a]();
    DW_OP_div[0x1b]();
    DW_OP_minus[0x1c]();
    DW_OP_mod[0x1d]();
    DW_OP_mul[0x1e]();
    DW_OP_neg[0x1f]();
    DW_OP_not[0x20]();
    DW_OP_or[0x21]();
    DW_OP_plus[0x22]();
    DW_OP_plus_uconst[0x22](addend: uleb128);
    DW_OP_shl[0x24]();
    DW_OP_shr[0x25]();
    DW_OP_shra[0x26]();
    DW_OP_xor[0x27]();
    DW_OP_bra[0x28](operand: 2u);
    DW_OP_eq[0x29]();
    DW_OP_ge[0x2a]();
    DW_OP_gt[0x2b]();
    DW_OP_le[0x2c]();
    DW_OP_lt[0x2d]();
    DW_OP_ne[0x2e]();
    DW_OP_skip[0x2f]();

    // DW_OP_list0..31
    DW_OP_lit0[0x30]();
    DW_OP_lit1[0x31]();
    DW_OP_lit2[0x32]();
    DW_OP_lit3[0x33]();
    DW_OP_lit4[0x34]();
    DW_OP_lit5[0x35]();
    DW_OP_lit6[0x36]();
    DW_OP_lit7[0x37]();
    DW_OP_lit8[0x38]();
    DW_OP_lit9[0x39]();
    DW_OP_lit10[0x3a]();
    DW_OP_lit11[0x3b]();
    DW_OP_lit12[0x3c]();
    DW_OP_lit13[0x3d]();
    DW_OP_lit14[0x3e]();
    DW_OP_lit15[0x3f]();
    DW_OP_lit16[0x40]();
    DW_OP_lit17[0x41]();
    DW_OP_lit18[0x42]();
    DW_OP_lit19[0x43]();
    DW_OP_lit20[0x44]();
    DW_OP_lit21[0x45]();
    DW_OP_lit22[0x46]();
    DW_OP_lit23[0x47]();
    DW_OP_lit24[0x48]();
    DW_OP_lit25[0x49]();
    DW_OP_lit26[0x4a]();
    DW_OP_lit27[0x4b]();
    DW_OP_lit28[0x4c]();
    DW_OP_lit29[0x4d]();
    DW_OP_lit30[0x4e]();
    DW_OP_lit31[0x4f]();

    // DW_OP_reg0..31
    DW_OP_reg0[0x50]();
    DW_OP_reg1[0x51]();
    DW_OP_reg2[0x52]();
    DW_OP_reg3[0x53]();
    DW_OP_reg4[0x54]();
    DW_OP_reg5[0x55]();
    DW_OP_reg6[0x56]();
    DW_OP_reg7[0x57]();
    DW_OP_reg8[0x58]();
    DW_OP_reg9[0x59]();
    DW_OP_reg10[0x5a]();
    DW_OP_reg11[0x5b]();
    DW_OP_reg12[0x5c]();
    DW_OP_reg13[0x5d]();
    DW_OP_reg14[0x5e]();
    DW_OP_reg15[0x5f]();
    DW_OP_reg16[0x60]();
    DW_OP_reg17[0x61]();
    DW_OP_reg18[0x62]();
    DW_OP_reg19[0x63]();
    DW_OP_reg20[0x64]();
    DW_OP_reg21[0x65]();
    DW_OP_reg22[0x66]();
    DW_OP_reg23[0x67]();
    DW_OP_reg24[0x68]();
    DW_OP_reg25[0x69]();
    DW_OP_reg26[0x6a]();
    DW_OP_reg27[0x6b]();
    DW_OP_reg28[0x6c]();
    DW_OP_reg29[0x6d]();
    DW_OP_reg30[0x6e]();
    DW_OP_reg31[0x6f]();

    // DW_breg_reg0..31
    DW_OP_bref0[0x70](offset: sleb128);
    DW_OP_bref1[0x71](offset: sleb128);
    DW_OP_bref2[0x72](offset: sleb128);
    DW_OP_bref3[0x73](offset: sleb128);
    DW_OP_bref4[0x74](offset: sleb128);
    DW_OP_bref5[0x75](offset: sleb128);
    DW_OP_bref6[0x76](offset: sleb128);
    DW_OP_bref7[0x77](offset: sleb128);
    DW_OP_bref8[0x78](offset: sleb128);
    DW_OP_bref9[0x79](offset: sleb128);
    DW_OP_bref10[0x7a](offset: sleb128);
    DW_OP_bref11[0x7b](offset: sleb128);
    DW_OP_bref12[0x7c](offset: sleb128);
    DW_OP_bref13[0x7d](offset: sleb128);
    DW_OP_bref14[0x7e](offset: sleb128);
    DW_OP_bref15[0x7f](offset: sleb128);
    DW_OP_bref16[0x80](offset: sleb128);
    DW_OP_bref17[0x81](offset: sleb128);
    DW_OP_bref18[0x82](offset: sleb128);
    DW_OP_bref19[0x83](offset: sleb128);
    DW_OP_bref20[0x84](offset: sleb128);
    DW_OP_bref21[0x85](offset: sleb128);
    DW_OP_bref22[0x86](offset: sleb128);
    DW_OP_bref23[0x87](offset: sleb128);
    DW_OP_bref24[0x88](offset: sleb128);
    DW_OP_bref25[0x89](offset: sleb128);
    DW_OP_bref26[0x8a](offset: sleb128);
    DW_OP_bref27[0x8b](offset: sleb128);
    DW_OP_bref28[0x8c](offset: sleb128);
    DW_OP_bref29[0x8d](offset: sleb128);
    DW_OP_bref30[0x8e](offset: sleb128);
    DW_OP_bref31[0x8f](offset: sleb128);

    DW_OP_regx[0x90](register: uleb128);
    DW_OP_fbreg[0x91](offset: sleb128);
    DW_OP_bregx[0x92](register: uleb128, offset: sleb128);
    DW_OP_piece[0x93](size: uleb128);
    DW_OP_deref_size[0x94](size: 1u);
    DW_OP_xderef_size[0x95](size: 1u);
    DW_OP_nop[0x96]();
    DW_OP_push_object_address[0x97]();
    DW_OP_call2[0x98](offset: 2u);
    DW_OP_call4[0x99](offset: 4u);
    // DW_OP_call_ref[0x9a](offset: dependent on dwarf bitness, not supported);
    DW_OP_form_tls_address[0x9b]();
    DW_OP_call_frame_cfa[0x9c]();
    DW_OP_bit_piece[0x9d](size: uleb128, offset: uleb128);
    // DW_OP_implicit_value takes a variable size immediate, not supported
    DW_OP_stack_value[0x9f]();
    // DWARF5 operations are somewhat more nuanced and I don't need them, not supported
}

macro_rules! define_cfa_ops {
    {
        $(
            $name: ident[$($sqargs: tt)*]($($pargs: tt)*);
        )*
    } => {
        $(define_cfa_ops!(@one $name[$($sqargs)*]($($pargs)*));)*
    };
    (@one $name: ident[$high: literal, $low: literal]($($operand_name: ident: $operand_type: tt),*)) => {
        pub fn $name(out: &mut Vec<u8> $(,
            $operand_name: define_dwarf_op!(@operand_param_type $operand_type)
        )*) {
            out.push(($high << 6) | $low);
            $(define_dwarf_op!(@push_operand $operand_type, out, $operand_name);)*
        }
    };
    (@one $name: ident[$high: literal, $low: ident]($($operand_name: ident: $operand_type: tt),*)) => {
        pub fn $name(out: &mut Vec<u8>, $low: u8 $(,
            $operand_name: define_dwarf_op!(@operand_param_type $operand_type)
        )*) {
            out.push(($high << 6) | $low);
            $(define_dwarf_op!(@push_operand $operand_type, out, $operand_name);)*
        }
    }
}

define_cfa_ops! {
    DW_CFA_advance_loc[1, delta]();
    DW_CFA_offset[2, register](offset: uleb128);
    DW_CFA_restore[3, register]();
    DW_CFA_nop[0, 0]();
    DW_CFA_set_loc[0, 0x01](address: address);
    DW_CFA_advance_loc1[0, 0x02](delta: 1u);
    DW_CFA_advance_loc2[0, 0x03](delta: 2u);
    DW_CFA_advance_loc4[0, 0x04](delta: 4u);
    DW_CFA_offset_extended[0, 0x05](register: uleb128, offset: uleb128);
    DW_CFA_restore_extended[0, 0x06](register: uleb128);
    DW_CFA_undefined[0, 0x07](register: uleb128);
    DW_CFA_same_value[0, 0x08](register: uleb128);
    DW_CFA_register[0, 0x09](register: uleb128, register2: uleb128);
    DW_CFA_remember_state[0, 0x0a]();
    DW_CFA_restore_state[0, 0x0b]();
    DW_CFA_def_cfa[0, 0x0c](register: uleb128, offset: uleb128);
    DW_CFA_def_cfa_register[0, 0x0d](register: uleb128);
    DW_CFA_def_cfa_offset[0, 0x0e](offset: uleb128);
    // these take a variable size block
    // DW_CFA_def_cfa_expression[0, 0x0f](BLOCK);
    // DW_CFA_expression[0, 0x10](register: uleb128, BLOCK);
    // DW_CFA_val_expression[0, 0x16](value: uleb128, BLOCK);
    DW_CFA_offset_extended_sf[0, 0x11](register: uleb128, offset: sleb128);
    DW_CFA_def_cfa_sf[0, 0x12](register: uleb128, offset: sleb128);
    DW_CFA_def_cfa_offset_sf[0, 0x13](offset: sleb128);
    DW_CFA_val_offset[0, 0x14](register: uleb128, offset: uleb128);
    DW_CFA_val_offset_sf[0, 0x15](register: uleb128, offset: sleb128);
}

macro_rules! define_dwarf_extended_registers {
    {
        $($name: ident = $number: literal;)*
    } => {
        $(#[inline] pub fn $name(out: &mut Vec<u8>) { DW_OP_regx(out, $number); })*
    };
}

pub mod x86_64 {
    #![allow(unused_imports)]

    use super::DW_OP_regx;

    define_dwarf_op!(DW_OP_addr, 0x06, adddress: 8u);
    pub use super::DW_OP_reg0 as DW_OP_reg_rax;
    pub use super::DW_OP_reg1 as DW_OP_reg_rdx;
    pub use super::DW_OP_reg2 as DW_OP_reg_rcx;
    pub use super::DW_OP_reg3 as DW_OP_reg_rbx;
    pub use super::DW_OP_reg4 as DW_OP_reg_rsi;
    pub use super::DW_OP_reg5 as DW_OP_reg_rdi;
    pub use super::DW_OP_reg6 as DW_OP_reg_rbp;
    pub use super::DW_OP_reg7 as DW_OP_reg_rsp;
    pub use super::DW_OP_reg8 as DW_OP_reg_r8;
    pub use super::DW_OP_reg9 as DW_OP_reg_r9;
    pub use super::DW_OP_reg10 as DW_OP_reg_r10;
    pub use super::DW_OP_reg11 as DW_OP_reg_r11;
    pub use super::DW_OP_reg12 as DW_OP_reg_r12;
    pub use super::DW_OP_reg13 as DW_OP_reg_r13;
    pub use super::DW_OP_reg14 as DW_OP_reg_r14;
    pub use super::DW_OP_reg15 as DW_OP_reg_r15;
    pub use super::DW_OP_reg16 as DW_OP_reg_return_address;
    pub use super::DW_OP_reg17 as DW_OP_reg_xmm0;
    pub use super::DW_OP_reg18 as DW_OP_reg_xmm1;
    pub use super::DW_OP_reg19 as DW_OP_reg_xmm2;
    pub use super::DW_OP_reg20 as DW_OP_reg_xmm3;
    pub use super::DW_OP_reg21 as DW_OP_reg_xmm4;
    pub use super::DW_OP_reg22 as DW_OP_reg_xmm5;
    pub use super::DW_OP_reg23 as DW_OP_reg_xmm6;
    pub use super::DW_OP_reg24 as DW_OP_reg_xmm7;
    pub use super::DW_OP_reg25 as DW_OP_reg_xmm8;
    pub use super::DW_OP_reg26 as DW_OP_reg_xmm9;
    pub use super::DW_OP_reg27 as DW_OP_reg_xmm10;
    pub use super::DW_OP_reg28 as DW_OP_reg_xmm11;
    pub use super::DW_OP_reg29 as DW_OP_reg_xmm12;
    pub use super::DW_OP_reg30 as DW_OP_reg_xmm13;
    pub use super::DW_OP_reg31 as DW_OP_reg_xmm14;
    define_dwarf_extended_registers! {
        DW_OP_reg_xmm15 = 32;
        DW_OP_reg_st0 = 33;
        DW_OP_reg_st1 = 34;
        DW_OP_reg_st2 = 35;
        DW_OP_reg_st3 = 36;
        DW_OP_reg_st4 = 37;
        DW_OP_reg_st5 = 38;
        DW_OP_reg_st6 = 39;
        DW_OP_reg_st7 = 40;
        DW_OP_reg_mm0 = 41;
        DW_OP_reg_mm1 = 42;
        DW_OP_reg_mm2 = 43;
        DW_OP_reg_mm3 = 44;
        DW_OP_reg_mm4 = 45;
        DW_OP_reg_mm5 = 46;
        DW_OP_reg_mm6 = 47;
        DW_OP_reg_mm7 = 48;
        DW_OP_reg_rFLAGS = 49;
        DW_OP_reg_es = 50;
        DW_OP_reg_cs = 51;
        DW_OP_reg_ss = 52;
        DW_OP_reg_ds = 53;
        DW_OP_reg_fs = 54;
        DW_OP_reg_gs = 55;
        DW_OP_reg_fs_base = 58;
        DW_OP_reg_gs_base = 59;
        DW_OP_reg_tr = 62;
        DW_OP_reg_ldtr = 63;
        DW_OP_reg_mxcsr = 64;
        DW_OP_reg_fcw = 65;
        DW_OP_reg_fsw = 66;
    }
}
