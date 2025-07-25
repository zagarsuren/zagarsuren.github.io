import { useEffect } from "react";
import { useLocation } from "react-router-dom";

export default function ScrollToTop() {
  const { pathname } = useLocation();

  useEffect(() => {
    // Always scroll to top when the route changes
    window.scrollTo(0, 0);
  }, [pathname]);

  return null;
}
