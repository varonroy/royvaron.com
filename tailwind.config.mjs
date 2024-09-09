/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
	theme: {
		extend: {
			colors: {
				link: 'rgb(29 78 216)'
			}
		},
	},
	plugins: [
		require('@tailwindcss/typography'),
	],
}
