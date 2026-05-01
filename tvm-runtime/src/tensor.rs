use core::ffi::c_void;
use std::ops::{Deref, DerefMut};
use std::ptr::null_mut;

use tvm_ffi::function_internal::ArgIntoRef;
use tvm_ffi::{
    collections::tensor::DLTensorExt as _, dtype::AsDLDataType, object::ObjectArc,
    type_traits::AnyCompatible, Any, AnyView, DLDataType, DLDevice, DLDeviceType, NDAllocator,
    Tensor as TVMFFITensor,
};
use tvm_ffi_sys::{DLTensor, TVMFFIAny};
use tvm_runtime_sys::{TVMDeviceAPIAllocDataSpace, TVMDeviceAPIFreeDataSpace, TVMDeviceAPIGet};

fn host_of_device(device: DLDevice) -> Option<DLDeviceType> {
    match device.device_type {
        DLDeviceType::kDLCPU => None,
        DLDeviceType::kDLCUDA => Some(DLDeviceType::kDLCUDAHost),
        DLDeviceType::kDLCUDAHost => None,
        DLDeviceType::kDLOpenCL => Some(DLDeviceType::kDLCPU),
        DLDeviceType::kDLVulkan => Some(DLDeviceType::kDLCPU),
        DLDeviceType::kDLMetal => Some(DLDeviceType::kDLCPU),
        DLDeviceType::kDLVPI => Some(DLDeviceType::kDLCPU),
        DLDeviceType::kDLROCM => Some(DLDeviceType::kDLROCMHost),
        DLDeviceType::kDLROCMHost => None,
        DLDeviceType::kDLExtDev => Some(DLDeviceType::kDLCPU),
        DLDeviceType::kDLCUDAManaged => Some(DLDeviceType::kDLCUDAHost),
        DLDeviceType::kDLOneAPI => Some(DLDeviceType::kDLCPU),
        DLDeviceType::kDLWebGPU => Some(DLDeviceType::kDLCPU),
        DLDeviceType::kDLHexagon => Some(DLDeviceType::kDLCPU),
        DLDeviceType::kDLMAIA => Some(DLDeviceType::kDLCPU),
        DLDeviceType::kDLTrn => Some(DLDeviceType::kDLCPU),
    }
}

fn is_host(device: DLDevice) -> bool {
    host_of_device(device).is_none()
}

#[derive(Debug, Clone)]
pub struct DeviceNDAlloc {}

unsafe impl Send for DeviceNDAlloc {}

unsafe impl Sync for DeviceNDAlloc {}

unsafe impl NDAllocator for DeviceNDAlloc {
    const MIN_ALIGN: usize = 64;

    unsafe fn alloc_data(&mut self, prototype: &DLTensor) -> *mut c_void {
        let numel = prototype.numel() as usize;
        let item_size = prototype.item_size();
        let size = numel * item_size as usize;
        let layout = std::alloc::Layout::from_size_align(size, Self::MIN_ALIGN).unwrap();
        TVMDeviceAPIAllocDataSpace(
            TVMDeviceAPIGet(prototype.device, false as i32),
            prototype.device,
            layout.size(),
            layout.align(),
            prototype.dtype,
        )
    }

    unsafe fn free_data(&mut self, tensor: &DLTensor) {
        TVMDeviceAPIFreeDataSpace(
            TVMDeviceAPIGet(tensor.device, false as i32),
            tensor.device,
            tensor.data,
        );
    }
}

#[repr(C)]
#[derive(Clone)]
pub struct Tensor {
    inner: TVMFFITensor,
}

impl Tensor {
    pub fn empty(shape: &[i64], dtype: DLDataType, device: DLDevice) -> Self {
        Self {
            inner: TVMFFITensor::from_nd_alloc(DeviceNDAlloc {}, shape, dtype, device),
        }
    }

    pub fn empty_like(other: &Tensor, device: DLDevice) -> Self {
        Self::empty(other.shape(), other.dtype(), device)
    }

    pub fn shape(&self) -> &[i64] {
        self.inner.shape()
    }

    pub fn dtype(&self) -> DLDataType {
        self.inner.dtype()
    }

    pub fn device(&self) -> DLDevice {
        self.inner.device()
    }

    /// Upstream tvm-ffi removed the public `dltensor()` accessor, but we still
    /// need a `DLTensor` for C-ABI calls like `TVMDeviceAPICopyDataFromTo`.
    /// Compose one from the public Tensor API. The resulting value is valid
    /// only as long as `self` is alive.
    pub fn dltensor(&self) -> DLTensor {
        DLTensor {
            data: self.inner.data_ptr() as *mut c_void,
            device: self.inner.device(),
            ndim: self.inner.ndim() as i32,
            dtype: self.inner.dtype(),
            shape: self.inner.shape().as_ptr() as *mut i64,
            strides: if self.inner.strides().is_empty() {
                core::ptr::null_mut()
            } else {
                self.inner.strides().as_ptr() as *mut i64
            },
            byte_offset: 0,
        }
    }

    pub fn dltensor_mut(&mut self) -> DLTensor {
        // Mutability in the C ABI sense — we still return by value because
        // the new tvm-ffi stores shape/strides through a private layout.
        self.dltensor()
    }

    pub fn data_ptr(&self) -> *const core::ffi::c_void {
        self.inner.data_ptr()
    }

    pub fn data_ptr_mut(&mut self) -> *mut core::ffi::c_void {
        self.inner.data_ptr_mut()
    }

    pub fn data_as_slice<T: AsDLDataType>(&self) -> tvm_ffi::error::Result<&[T]> {
        self.inner.data_as_slice()
    }

    pub fn data_as_slice_mut<T: AsDLDataType>(&mut self) -> tvm_ffi::error::Result<&mut [T]> {
        self.inner.data_as_slice_mut()
    }

    pub fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    pub fn numel(&self) -> usize {
        self.inner.numel()
    }

    pub fn strides(&self) -> &[i64] {
        self.inner.strides()
    }

    pub fn copy_from(&mut self, src: &Self) -> tvm_ffi::error::Result<()> {
        if is_host(self.device()) && is_host(src.device()) {
            // Host to Host copy
            let numel = self.shape().iter().product::<i64>() as usize;
            let item_size = (self.dtype().bits / 8) as usize;
            let nbytes = numel * item_size;

            let src_data = src.data_as_slice::<u8>()?;
            debug_assert_eq!(
                nbytes,
                src_data.len(),
                "data length ({}) does not match tensor size ({})",
                src_data.len(),
                nbytes
            );
            let src_ptr = src_data.as_ptr() as *const u8;
            let dst_ptr = self.data_as_slice_mut()?.as_mut_ptr() as *mut u8;
            unsafe {
                core::ptr::copy_nonoverlapping(src_ptr, dst_ptr, nbytes);
            }
        } else {
            let device = if is_host(self.device()) {
                // Device to Host copy
                src.device()
            } else {
                // Host to Device copy, or
                // Device to Device copy
                self.device()
            };
            let src_dl = src.dltensor();
            let mut dst_dl = self.dltensor_mut();
            unsafe {
                let handle = TVMDeviceAPIGet(device, false as i32);
                tvm_runtime_sys::TVMDeviceAPICopyDataFromTo(
                    handle,
                    &src_dl,
                    &mut dst_dl,
                    null_mut(),
                );
                tvm_runtime_sys::TVMDeviceAPIStreamSync(handle, device, null_mut());
                tvm_runtime_sys::TVMDeviceAPIDestroy(handle);
            }
        }
        Ok(())
    }

    pub fn copy_from_slice(&mut self, src: &[u8]) -> tvm_ffi::error::Result<()> {
        if is_host(self.device()) {
            // Host to Host copy
            let numel = self.shape().iter().product::<i64>() as usize;
            let item_size = (self.dtype().bits / 8) as usize;
            let nbytes = numel * item_size;

            debug_assert_eq!(
                nbytes,
                src.len(),
                "data length ({}) does not match tensor size ({})",
                src.len(),
                nbytes
            );
            let src_ptr = src.as_ptr() as *const u8;
            let dst_ptr = self.data_as_slice_mut()?.as_mut_ptr() as *mut u8;
            unsafe {
                core::ptr::copy_nonoverlapping(src_ptr, dst_ptr, nbytes);
            }
        } else {
            // Host to Device copy
            let self_dl = self.dltensor();
            let host_dltensor = DLTensor {
                data: src.as_ptr() as *mut c_void,
                device: DLDevice {
                    device_type: host_of_device(self.device())
                        .expect("The matching host device type should be exist"),
                    device_id: 0,
                },
                ndim: self_dl.ndim,
                dtype: self_dl.dtype,
                shape: self_dl.shape,
                strides: self_dl.strides,
                byte_offset: self_dl.byte_offset,
            };
            let mut dst_dl = self.dltensor_mut();
            unsafe {
                let handle = TVMDeviceAPIGet(self.device(), false as i32);
                tvm_runtime_sys::TVMDeviceAPICopyDataFromTo(
                    handle,
                    &host_dltensor,
                    &mut dst_dl,
                    null_mut(),
                );
                tvm_runtime_sys::TVMDeviceAPIStreamSync(handle, self.device(), null_mut());
                tvm_runtime_sys::TVMDeviceAPIDestroy(handle);
            }
        }

        Ok(())
    }

    pub fn reshape(&self, shape: &[i64]) -> tvm_ffi::error::Result<Tensor> {
        let nelem_before = self.shape().iter().product::<i64>();
        let nelem_after = shape.iter().product::<i64>();
        if nelem_before != nelem_after {
            tvm_ffi::bail!(
                tvm_ffi::VALUE_ERROR,
                "Cannot reshape from {:?} to {:?}",
                self.shape(),
                shape,
            );
        }

        let mut new_tensor = Self::empty(shape, self.dtype(), self.device());
        new_tensor.copy_from(self).unwrap();

        Ok(new_tensor)
    }
}

impl From<TVMFFITensor> for Tensor {
    fn from(inner: TVMFFITensor) -> Self {
        Self { inner }
    }
}

impl From<Tensor> for TVMFFITensor {
    fn from(value: Tensor) -> Self {
        value.inner
    }
}

impl AsRef<TVMFFITensor> for Tensor {
    fn as_ref(&self) -> &TVMFFITensor {
        return &self.inner;
    }
}

impl AsMut<TVMFFITensor> for Tensor {
    fn as_mut(&mut self) -> &mut TVMFFITensor {
        &mut self.inner
    }
}

impl Deref for Tensor {
    type Target = TVMFFITensor;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Tensor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

unsafe impl AnyCompatible for Tensor {
    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        TVMFFITensor::copy_to_any_view(&src.inner, data);
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        TVMFFITensor::move_to_any(src.inner, data);
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        TVMFFITensor::check_any_strict(data)
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        Self {
            inner: TVMFFITensor::copy_from_any_view_after_check(data),
        }
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        Self {
            inner: TVMFFITensor::move_from_any_after_check(data),
        }
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        TVMFFITensor::try_cast_from_any_view(data).map(|inner| Self { inner })
    }

    fn type_str() -> String {
        TVMFFITensor::type_str()
    }
}

unsafe impl tvm_ffi::object::ObjectRefCore for Tensor {
    type ContainerType = <TVMFFITensor as tvm_ffi::object::ObjectRefCore>::ContainerType;

    fn data(this: &Self) -> &ObjectArc<Self::ContainerType> {
        TVMFFITensor::data(&this.inner)
    }

    fn into_data(this: Self) -> ObjectArc<Self::ContainerType> {
        TVMFFITensor::into_data(this.inner)
    }

    fn from_data(data: ObjectArc<Self::ContainerType>) -> Self {
        Self {
            inner: TVMFFITensor::from_data(data),
        }
    }
}

impl<'a> TryFrom<AnyView<'a>> for Tensor {
    type Error = <TVMFFITensor as TryFrom<AnyView<'a>>>::Error;

    #[inline(always)]
    fn try_from(value: AnyView<'a>) -> Result<Self, Self::Error> {
        <TVMFFITensor as TryFrom<AnyView<'a>>>::try_from(value).map(|t| t.into())
    }
}

impl TryFrom<Any> for Tensor {
    type Error = <TVMFFITensor as TryFrom<Any>>::Error;

    #[inline(always)]
    fn try_from(value: Any) -> Result<Self, Self::Error> {
        <TVMFFITensor as TryFrom<Any>>::try_from(value).map(|t| t.into())
    }
}

impl ArgIntoRef for Tensor {
    type Target = <TVMFFITensor as ArgIntoRef>::Target;

    fn to_ref(&self) -> &Self::Target {
        <TVMFFITensor as ArgIntoRef>::to_ref(self)
    }
}
