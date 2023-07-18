/** @type {import('tailwindcss').Config} */
module.exports = {
	content: [
		"./html-generated/**/*.html",
		"./js/**/*.js",
	],
	theme: {
		extend: {
			colors: {
				link: 'rgb(29 78 216)'
			}
		}
	},
	plugins: [
		require('@tailwindcss/typography')
	],
}
