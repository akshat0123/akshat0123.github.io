import Sidebar from "../components/sidebar";
import Header from "../components/header";
import Layout from "../components/layout";
import RData from "../content/research.json";
import React from "react";

export default () => (
	<div>
		<Header/>
		<Layout>
			<h1>Research</h1>
			<ul class="nobullet">
				{RData.papers.map((data, index) => {
					return <ul class="nobullet">
						<li><a href={data.link}><b>{data.title}</b></a></li>
                        <li>{data.authors}</li>
						<li><i>{data.location}</i></li>
						<br/>
					</ul>
				})}
			</ul>
		</Layout>
		<Sidebar/>
	</div>
)
