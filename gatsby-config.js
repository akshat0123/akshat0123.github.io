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
                      options: {
                          // Add any KaTeX options from https://github.com/KaTeX/KaTeX/blob/master/docs/options.md here
                          strict: `ignore`
                      }
                  },
                  {
                    resolve: `gatsby-remark-images`,
                    options: {
                        maxWidth: 800
                    }
                  }
              ],
            },
        },
        `gatsby-plugin-sharp`
    ]
}
