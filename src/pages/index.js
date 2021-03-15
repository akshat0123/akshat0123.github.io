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
                  Software Engineer with a focus on machine learning and data science application development, including work with Python, NumPy, PyTorch, SQL, GCP, and AWS – in addition to a Master’s degree. Available to relocate nationwide.
				</p>
				<h1>Interests</h1>
				<p>
					Machine Learning, Natural Language Processing, Bias and Fairness in AI
				</p>
			</div>
		</Layout>
		<Sidebar/>
	</div>
)
