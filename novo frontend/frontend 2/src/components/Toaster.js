/* ========================================
 * Toaster Component
 * Wrapper do Sonner toast com tema dark/light
 * ======================================== */

import {
  u as useThemeStore,
  j as jsx,
  d as defaultConfig,
} from "/src/core/main.js";
import { T as SonnerToaster } from "/src/primitives/toastRuntime.js";

function Toaster() {
  const { isDark, notification } = useThemeStore();
  return jsx.jsx(SonnerToaster, {
    theme: isDark ? "dark" : "light",
    offset: "16px",
    position: notification?.position || defaultConfig?.notification?.position,
    expand: notification?.isExpanded || defaultConfig?.notification?.isExpanded,
    visibleToasts:
      notification?.visibleToasts || defaultConfig?.notification?.visibleToasts,
    richColors: true,
  });
}

export { Toaster as default };
