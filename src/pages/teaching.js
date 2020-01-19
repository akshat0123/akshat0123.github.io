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
			<div>
				<h1><Typist cursor={{hideWhenDone:true, hideWhenDoneDelay:250}}>Teaching</Typist></h1>
				<table>
					<tr>
						<th>Course</th>
						<th>Role</th>
						<th>University</th>
						<th>Dates</th>
					</tr>
					{CVData.teaching.map((data, idx) => {
						return <tr>
							<td>{data.course}</td>
							<td>{data.role}</td>
							<td>{data.university}</td>
							<td>{data.dates}</td>
						</tr>
					})}
				</table>
			</div>
		</Layout>
		<Sidebar/>
	</div>
)
