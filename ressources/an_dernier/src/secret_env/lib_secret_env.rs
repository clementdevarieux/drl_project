use libloading::Library;
use once_cell::sync::Lazy;

pub fn getlib() -> Library {
    unsafe {
        #[cfg(target_os = "linux")]
            let path = "./libs/libsecret_envs.so";
        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
            let path = "./libs/libsecret_envs_intel_macos.dylib";
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let path = "./libs/libsecret_envs.dylib";
        #[cfg(windows)]
            let path = "./libs/secret_envs.dll";
        let lib = libloading::Library::new(path).expect("Failed to load library");
        return lib
    }
}

pub static LIB: Lazy<Library> = Lazy::new(|| {
    getlib()
});