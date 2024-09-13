// @ts-check
import { defineConfig } from "astro/config";

import tailwind from "@astrojs/tailwind";

// https://github.com/remarkjs/remark-math
import remarkMath from "remark-math";

// https://github.com/remarkjs/remark-math/tree/main/packages/rehype-katex
// https://r3zz.io/posts/astro-blog-latex/
import rehypeKatex from "rehype-katex";

// https://www.npmjs.com/package/rehype-mathjax
// https://github.com/remarkjs/remark-math?tab=readme-ov-file#example-mathjax
// import rehypeMathjax from "rehype-mathjax";

import mdx from "@astrojs/mdx";

// https://astro.build/config
export default defineConfig({
  integrations: [tailwind(), mdx()],
  // https://docs.astro.build/en/guides/markdown-content/#markdown-plugins
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    // rehypePlugins: [rehypeMathjax],
  },
});
