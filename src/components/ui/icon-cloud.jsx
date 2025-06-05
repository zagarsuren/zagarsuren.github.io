"use client";

import { useEffect, useMemo, useState } from "react";
import { useTheme } from "next-themes";
import { fetchSimpleIcons, renderSimpleIcon } from "react-icon-cloud";

// Custom icon renderer with adjustable size
export const renderCustomIcon = (icon, theme, size = 64) => {
  const bgHex = theme === "light" ? "#f3f2ef" : "#080510";
  const fallbackHex = theme === "light" ? "#6e6e73" : "#ffffff";
  const minContrastRatio = theme === "dark" ? 2 : 1.2;

  return (
    <div
      key={icon.slug}
      style={{ width: size, height: size }}
      className="flex items-center justify-center"
      title={icon.title}
    >
      {renderSimpleIcon({
        icon,
        bgHex,
        fallbackHex,
        minContrastRatio,
        size: size * 0.7, // scale inner SVG slightly smaller than container
        aProps: {
          href: undefined,
          target: undefined,
          rel: undefined,
          onClick: (e) => e.preventDefault(),
        },
      })}
    </div>
  );
};

export default function IconCloud({
  iconSlugs = [],        // array of simple-icon slugs
  imageArray = [],       // array of image URLs
  size = 64,             // icon size in pixels
}) {
  const [data, setData] = useState(null);
  const { theme } = useTheme();

  useEffect(() => {
    if (iconSlugs.length > 0) {
      fetchSimpleIcons({ slugs: iconSlugs }).then(setData);
    }
  }, [iconSlugs]);

  const renderedIcons = useMemo(() => {
    if (!data) return null;

    return Object.values(data.simpleIcons).map((icon) =>
      renderCustomIcon(icon, theme || "dark", size)
    );
  }, [data, theme, size]);

  return (
    <div className="w-full py-8 px-4">
      <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-10 gap-6 justify-items-center">
        {renderedIcons}

        {imageArray.length > 0 &&
          imageArray.map((image, index) => (
            <div
              key={`img-${index}`}
              className="flex items-center justify-center"
              style={{ width: size, height: size }}
            >
              <img
                src={image}
                alt={`icon-${index}`}
                height={size}
                width={size}
                className="object-contain"
              />
            </div>
          ))}
      </div>
    </div>
  );
}
