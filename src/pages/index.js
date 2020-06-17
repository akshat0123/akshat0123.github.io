import Sidebar from "../components/sidebar";
import Header from "../components/header";
import Layout from "../components/layout";
import React from "react";

export default () => (
	<div>
		<Header/>
		<Layout>
			<div>
				<h1>About Me</h1>
				<p>
					I am a PhD student in computer science at George Washington
					University. My advisor is Professor Aylin Caliskan and my
					primary research interest is in bias and fairness in machine
					learning.
				</p>
				<h1>Research</h1>
				<p>
					Bias and Fairness in AI, Natural Language Processing,
					Machine Learning
				</p>
			</div>
		</Layout>
		<Sidebar/>
	</div>
)
