import { defineConfig } from "vitest/config";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { svelteTesting } from "@testing-library/svelte/vite";

// Component tests run in jsdom. `svelteTesting()` swaps in Svelte's browser
// build, runs the per-test setup, and cleans up the DOM after each test.
export default defineConfig({
  plugins: [svelte(), svelteTesting()],
  test: {
    environment: "jsdom",
    include: ["src/**/*.{test,spec}.ts"],
    setupFiles: ["./vitest-setup.ts"],
  },
});
