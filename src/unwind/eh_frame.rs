use crate::{sleb128, uleb128, DW_CFA_nop};

use super::_Unwind_Personality_Fn;

fn dwarf_align(data: &mut Vec<u8>, start: usize) {
    let real_length = data.len() - start;
    let aligned_length = real_length.next_multiple_of(8);
    data[start..start + 4].copy_from_slice(&((aligned_length - 4) as u32).to_le_bytes());
    for _ in real_length..aligned_length {
        DW_CFA_nop(data);
    }
}

pub struct EhFrameBuilder {
    out: Vec<u8>,
}

impl EhFrameBuilder {
    pub fn new(
        code_alignment_factor: u64,
        data_alignment_factor: i64,
        eh_personality: _Unwind_Personality_Fn,
        instruction_builder: impl FnOnce(&mut Vec<u8>),
    ) -> Self {
        let mut data = vec![];

        // CIE
        {
            // length
            data.extend([0u8; 4]);
            // CIE_id
            data.extend(0u32.to_le_bytes());
            // version
            data.push(1);
            // augmentation
            data.extend(b"zP\0");
            // code_alignment_factor
            uleb128(&mut data, code_alignment_factor);
            // data_alignment_factor
            sleb128(&mut data, data_alignment_factor);
            // return_address_register
            data.push(16);
            // leb128(&mut data, /* "Return adddress RA" */ 16);
            // optional CIE augmentation section
            uleb128(&mut data, 9);
            data.push(0x00); // absolute pointer
            data.extend((eh_personality as u64).to_le_bytes());
            // initial_instructions
            instruction_builder(&mut data);
        }

        dwarf_align(&mut data, 0);

        Self { out: data }
    }

    pub fn add_fde(
        mut self,
        initial_location: u64,
        address_range: u64,
        instruction_builder: impl FnOnce(&mut Vec<u8>),
    ) -> Self {
        let fde_start = self.out.len();
        // length
        self.out.extend([0u8; 4]);
        // CIE_pointer
        self.out.extend((self.out.len() as u32).to_le_bytes());
        // initial_location
        self.out.extend(initial_location.to_le_bytes());
        // address_range
        self.out.extend(address_range.to_le_bytes());
        // fde augmentation section
        uleb128(&mut self.out, 0);
        // call frame instructions
        instruction_builder(&mut self.out);
        dwarf_align(&mut self.out, fde_start);
        self
    }

    pub fn build(mut self) -> Vec<u8> {
        // libgcc detects the end of an .eh_frame section via the presence of a zero-length FDE
        // see https://github.com/llvm/llvm-project/blob/1055c5e1d316164c70e0c9f016411a28f3b4792e/llvm/lib/ExecutionEngine/Orc/TargetProcess/RegisterEHFrames.cpp#L128
        self.out.extend([0, 0, 0, 0]);
        self.out
    }
}

// This will be useful when supporting non-libgcc unwinders
// (i.e. libunwind which doesn't accept whole .eh_frame sections but single FDEs)
#[allow(dead_code)]
fn walk_eh_frame_fdes(eh_frame: &[u8], callback: impl Fn(usize)) {
    let mut offset = 0;
    while offset < eh_frame.len() {
        let current = &eh_frame[offset..];
        let len = u32::from_le_bytes(current[0..4].try_into().unwrap()) as usize;
        if len == 0 {
            offset += 4;
            break;
        }
        let cie_pointer = u32::from_le_bytes(current[4..8].try_into().unwrap());
        if cie_pointer == 0 {
            // is a cie
        } else {
            callback(offset);
        }
        offset += len + 4;
    }
    assert_eq!(offset, eh_frame.len());
}
