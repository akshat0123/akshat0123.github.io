import Sidebar from "../components/sidebar";
import Header from "../components/header";
import Layout from "../components/layout";
import CVData from "../content/cv.json";
import Typist from "react-typist"
import React from "react";

export default () => (
	<div>
		<Header/>
		<Layout>
			<h1><Typist cursor={{hideWhenDone:true, hideWhenDoneDelay:250}}>Teaching</Typist></h1>
			<ul class="nobullet">
				{CVData.teaching.map((data, index) => {
					return <ul class="nobullet">
						<li><b>{data.course}</b></li>
						<li><i>{data.role}</i></li>
						<li>{data.university}</li>
						<li>{data.dates}</li>
						<br/>
					</ul>
				})}
			</ul>
		</Layout>
		<Sidebar/>
	</div>
)
