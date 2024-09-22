#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]
#![allow(improper_ctypes_definitions)]

use cuda_std::prelude::*;
use cust::prelude::*;
use cust::DeviceCopy;
use cuda_std::thread::{block_dim_x, block_idx_x, thread_idx_x};
use byteorder::{ByteOrder, LittleEndian};

// #[panic_handler]
// fn panic(_: &::core::panic::PanicInfo) -> ! {
//     unsafe { trap() }
// }


extern crate alloc;

#[inline]
fn add_u256(base: &[u8; 32], mut offset: u64) -> [u8; 32] {
    let mut res = [0; 32];
    for i in 0..4 {
        let start = i * 8;
        let end = (i + 1) * 8;
        let base = LittleEndian::read_u64(&base[start..end]);
        let (total, overflow) = base.overflowing_add(offset);
        LittleEndian::write_u64(&mut res[start..end], total);
        if overflow {
            offset = 1;
        } else {
            offset = 0;
        }
    }
    res
}


#[kernel]
pub unsafe fn render(params_ptr: *mut KernelParams) {
    let params = unsafe { &mut *params_ptr };
    let x = unsafe { block_dim_x() * block_idx_x() + thread_idx_x() } as u64;

    let seed = unsafe { core::slice::from_raw_parts(params.seed.as_ptr(), 32) }
        .try_into()
        .unwrap();
    let cur_seed = add_u256(seed, x);
    let s = ed25519_compact::Seed::new(cur_seed);
    let kp = ed25519_compact::KeyPair::from_seed(s);

    let byte_prefixes =
        unsafe { core::slice::from_raw_parts_mut(params.byte_prefixes.as_mut_ptr(), params.byte_prefixes_len) };
    for byte_prefix in byte_prefixes {
        if byte_prefix.matches(&*kp.pk) {
            let out = unsafe { core::slice::from_raw_parts_mut(byte_prefix.out.as_mut_ptr(), 32) };
            out.clone_from_slice(&cur_seed);
            let success = unsafe { &mut *byte_prefix.success.as_mut_ptr() };
            *success = true;
        }
    }
}

#[derive(DeviceCopy, Copy, Clone)]
#[repr(C)]
pub struct KernelParams {
    pub seed: DevicePointer<u8>,
    pub byte_prefixes: DevicePointer<BytePrefix>,
    pub byte_prefixes_len: usize,
}

#[derive(DeviceCopy, Copy, Clone)]
#[repr(C)]
pub struct BytePrefix {
    pub byte_prefix: DevicePointer<u8>,
    pub byte_prefix_len: usize,
    pub last_byte_idx: usize,
    pub last_byte_mask: u8,
    pub out: DevicePointer<u8>,
    pub success: DevicePointer<bool>,
}
impl BytePrefix {
    pub fn matches(&self, data: &[u8]) -> bool {
        let slice =
            unsafe { core::slice::from_raw_parts(self.byte_prefix.as_ptr(), self.byte_prefix_len) };
        data.starts_with(&slice[..self.last_byte_idx])
            && data[self.last_byte_idx] & self.last_byte_mask == slice[self.last_byte_idx]
    }
}
