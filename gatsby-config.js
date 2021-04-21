/**
 * Configure your Gatsby site with this file.
 *
 * See: https://www.gatsbyjs.org/docs/gatsby-config/
 */

module.exports = {
    /* Your site config here */
    plugins: [
        {
            resolve: `gatsby-source-filesystem`,
            options: {
                name: `markdown-pages`,
                path: `${__dirname}/src/content/blogposts`,
            },
        },
        {
            resolve: `gatsby-transformer-remark`,
            options: {
              plugins: [
                  {
                      resolve: `gatsby-remark-katex`,
                      options: { strict: `ignore` }
                  },
                  {
                    resolve: `gatsby-remark-images`,
                    options: { maxWidth: 900 }
                  },
                  {
                    resolve: `gatsby-remark-highlight-code`,
                    options: { theme: `one-light` }
                  }
              ],
            },
        },
        `gatsby-plugin-sharp`,
        {
            resolve: `gatsby-plugin-google-gtag`,
            options: {
                trackingIds: [
                    "G-M8Y04F3FJN"
                ]
            }
        }
    ]
}
