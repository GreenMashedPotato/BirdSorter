fn main() {
    // Bake the build date into the binary for the About dialog.
    let now = time::OffsetDateTime::now_utc();
    println!(
        "cargo:rustc-env=BUILD_DATE={:04}-{:02}-{:02}",
        now.year(),
        now.month() as u8,
        now.day()
    );
    tauri_build::build()
}
