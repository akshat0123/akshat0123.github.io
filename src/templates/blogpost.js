import { defineCustomElements as deckDeckGoHighlightElement } from '@deckdeckgo/highlight-code/dist/loader';
import Header from "../components/header";
import { graphql } from "gatsby"
import "katex/dist/katex.min.css"
import React from "react"

export default function Template({ data }) {
    deckDeckGoHighlightElement();
    const { markdownRemark } = data // data.markdownRemark holds your post data
    const { frontmatter, html } = markdownRemark
    return (
        <div id="blogcontainer" className="blog-post-container">
            <Header/>
            <div id="blogpost">
                <h1>{frontmatter.title}</h1>
                <div className="blog-post-content" dangerouslySetInnerHTML={{ __html: html }}/>
            </div>
        </div>
    )
}

export const pageQuery = graphql`
  query($slug: String!) {
    markdownRemark(frontmatter: { slug: { eq: $slug } }) {
      html
      frontmatter {
        slug
        title
      }
    }
  }
`
