import Projects from "../content/projects.json";
import Sidebar from "../components/sidebar";
import Header from "../components/header";
import Layout from "../components/layout";
import React from "react";

export default () => (
	<div>
		<Header/>
		<Layout>
			<h1>Research</h1>
			<ul className="nobullet">
				{Projects.research.map((data, index) => {
					return (<ul className="nobullet">
						<li><a href={data.link}><b>{data.title}</b></a></li>
                        <li>{data.authors}</li>
                        <li><i>{data.location}</i></li>
						<br/>
					</ul>)
				})}
			</ul>
		</Layout>
		<Sidebar/>
	</div>
)
