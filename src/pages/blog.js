import Sidebar from "../components/sidebar";
import Layout from "../components/layout";
import Header from "../components/header";
import { Link, graphql } from "gatsby";
import React from "react"

export default ({ data: { allMarkdownRemark: { groups } }}) => {

    return (
        <div>
            <Header/>
            <Layout>
                <p>
                    Some brief notes/tutorials I wrote regarding different machine learning algorithms/approaches. 
                    Feel free to reach out with any questions/concerns/corrections.
                </p>
                {groups.map((group, i) => {
                    return (
                        <div>
                        <h2>{group.category}</h2>
                        <ul className="nobullet">{group.posts.map((data, j) => {
                            return (<li>
                                <Link to={data.post.slug}>
                                    {data.post.title}
                                </Link>
                            </li>)
                        })}</ul>
                        </div>
                    )
                })}
            </Layout>
            <Sidebar/>
        </div>
    )
}

export const pageQuery = graphql` {
    allMarkdownRemark(
        sort: {
            fields: [frontmatter___order],
            order: ASC
        }
    ){
        groups: group(field: frontmatter___category){
            category: fieldValue
            posts: nodes {
                post: frontmatter {
                    title
                    slug
                }
            }
        }
    }
}
`
